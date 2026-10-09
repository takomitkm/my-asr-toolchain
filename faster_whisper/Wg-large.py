import os
import re
import logging
import logging.handlers
import shutil
from pathlib import Path
from datetime import datetime
from send2trash import send2trash
from faster_whisper import WhisperModel, BatchedInferencePipeline
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dir(env_name, default_rel):
    val = (os.environ.get(env_name) or "").strip()
    val = os.path.expanduser(os.path.expandvars(val))
    if not os.path.isabs(val):
        val = os.path.join(PKG_DIR, val or default_rel)
    return val


DEVICE = "cuda"
DEVICE_INDEX = 0
COMPUTE_TYPE = "int8_float16"
LOCAL_FILES_ONLY = True
MODEL_PATH = resolve_dir("FASTER_WHISPER_MODEL_DIR", "model")
BATCH_SIZE = 10

ASR_ROOT = Path(resolve_dir("ASR_TEXT_DIR", "txt"))
ASRSOURCE = Path(resolve_dir("ASR_AUDIO_DIR", "audio"))
DOWNLOAD_FOLDER = ASRSOURCE / "converted"
RULES_FILE = ASR_ROOT / "rules.txt"
LOG_FOLDER = ASR_ROOT / "log"
FAILED_FOLDER = ASRSOURCE / "failed"
TEMPTXT = ASR_ROOT / "ear.txt"
ARCHIVE_TXT = ASR_ROOT / "a.txt"

LANGUAGE = None
EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".oga", ".ogg", ".opus"}
INITIAL_PROMPT = ""

for p in [ASR_ROOT, LOG_FOLDER, FAILED_FOLDER]:
    p.mkdir(parents=True, exist_ok=True)

log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_FOLDER / f"whisper_{log_timestamp}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
console_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

model = None
batched_model = None

def load_rules(path: Path):
    rules_list = []
    if not path.exists():
        logger.info(f"规则文件不存在：{path}")
        return rules_list
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if " = " in line:
                    parts = line.split(" = ", 1)
                    rules_list.append((re.compile(parts[0].strip()), parts[1].strip()))
        logger.info(f"加载规则 {len(rules_list)} 条")
    except Exception as e:
        logger.error(f"Failed to load rules: {e}")
    return rules_list

def apply_rules(text, rules_list):
    for pattern, repl in rules_list:
        text = pattern.sub(repl, text)
    return text

def find_oldest_audio():
    files = [f for f in DOWNLOAD_FOLDER.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS]
    if files:
        oldest = min(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"找到音频文件：{oldest}")
        return oldest
    else:
        logger.info(f"在 {DOWNLOAD_FOLDER} 及其子目录中未找到 .wav 文件")
        return None

def transcribe_audio(file_path: Path, rules_list: list):
    try:
        segments, info = batched_model.transcribe(
            str(file_path),
            batch_size=BATCH_SIZE,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=INITIAL_PROMPT,
            language=LANGUAGE
        )
        lines = [f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}" for seg in segments]
        full_text = "\n".join(lines) + "\n"
        return apply_rules(full_text, rules_list), None
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return None, str(e)

def process_audio_files(rules_list: list):
    logger.info("开始处理音频文件")
    error_log = []
    while True:
        source_file = find_oldest_audio()
        if source_file:
            output_text, error = transcribe_audio(source_file, rules_list)
            if error or not output_text:
                error_msg = f"[Errno {source_file.name}] {error}"
                logger.critical(f"FAILED: {error_msg}")
                error_log.append((source_file.name, error))
                try:
                    shutil.move(str(source_file), str(FAILED_FOLDER / source_file.name))
                    logger.info(f"已移动失败文件至：{FAILED_FOLDER}")
                except Exception as move_err:
                    logger.error(f"移动文件失败：{move_err}")
                continue
            with open(TEMPTXT, "a", encoding="utf-8") as f:
                f.write(f"title:{source_file.relative_to(DOWNLOAD_FOLDER)}\n{output_text}\n\n")
            logger.info(f"转写完成：{source_file.name}")
            try:
                send2trash(str(source_file))
                logger.info(f"已删除：{source_file}")
            except Exception as e:
                logger.error(f"Trash error for {source_file.name}: {e}")
        else:
            logger.info("无更多音频文件，退出循环")
            break
    
    if error_log:
        logger.critical("--- 错误汇总 ---")
        for fname, err in error_log:
            logger.critical(f"文件：{fname} | 错误：{err}")
        logger.critical(f"总计失败文件数：{len(error_log)}")
    else:
        logger.info("所有文件处理成功，无错误")
        
    return True

def handle_no_file():
    if TEMPTXT.exists() and TEMPTXT.stat().st_size > 0:
        content = TEMPTXT.read_text(encoding="utf-8")
        try:
            with open(ARCHIVE_TXT, "a", encoding="utf-8") as f:
                f.write(f"\n--- Archive at {datetime.now()} ---\n{content}")
            send2trash(str(TEMPTXT))
            logger.info("ear.txt 已归档并删除")
        except Exception as e:
            logger.error(f"Archive error: {e}")
    else:
        logger.info("ear.txt 为空或不存在")

def main_loop():
    global model, batched_model
    logger.info("Starting main_loop")
    rules = load_rules(RULES_FILE)
    try:
        logger.info(f"加载模型：{MODEL_PATH}")
        model = WhisperModel(
            MODEL_PATH,
            device=DEVICE,
            device_index=DEVICE_INDEX,
            compute_type=COMPUTE_TYPE,
            local_files_only=LOCAL_FILES_ONLY
        )
        logger.info("模型加载成功")
        batched_model = BatchedInferencePipeline(model)
        logger.info("Pipeline 创建成功")
    except Exception as e:
        logger.critical(f"Initialization Failed: {e}", exc_info=True)
        return

    process_audio_files(rules)
    # handle_no_file()
    logger.info("脚本执行完毕")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.critical(f"未捕获异常：{e}", exc_info=True)
    finally:
        logging.shutdown()