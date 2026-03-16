import asyncio
import wave
import struct
import math
import os
import re
import pyautogui
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
WAV_PATH = r"D:\harvard.wav"

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

# ---------------------------------------------------------------------------
# ANIMATION BRIDGE — Logic Tag → Virtual Key Mapping
# ---------------------------------------------------------------------------
# Các tag này được AI chèn vào đầu mỗi câu trả lời.
# Bạn map các phím này sang animation tương ứng trong UE5 Blueprint.
#
#   [IDLE]    → phím N  — Trạng thái đứng yên / kết thúc nói
#   [SPEAK]   → phím M  — Nói chuyện thông thường (default khi có audio)
#   [WAVE]    → phím 1  — Vẫy tay chào hỏi / tạm biệt
#   [NOD]     → phím 2  — Gật đầu đồng ý / xác nhận
#   [POINT]   → phím 3  — Chỉ tay (hướng dẫn, giới thiệu địa điểm)
#   [THINK]   → phím 4  — Tư duy / cân nhắc (tay chạm cằm)
#   [SHRUG]   → phím 5  — Nhún vai (không biết / không chắc)
#   [EXCITED] → phím 6  — Hứng khởi / vui mừng (sự kiện, tin tốt)
# ---------------------------------------------------------------------------
LOGIC_TAG_KEY_MAP = {
    "[IDLE]":    "n",
    "[SPEAK]":   "m",
    "[WAVE]":    "1",
    "[NOD]":     "2",
    "[POINT]":   "3",
    "[THINK]":   "4",
    "[SHRUG]":   "5",
    "[EXCITED]": "6",
}

# Regex để tìm tag trong transcript
TAG_PATTERN = re.compile(r'\[(IDLE|SPEAK|WAVE|NOD|POINT|THINK|SHRUG|EXCITED)\]')


def parse_and_trigger_gesture(text: str) -> str | None:
    """
    Tìm logic tag trong transcript của AI.
    Ấn phím gesture tương ứng nếu tìm thấy tag đặc biệt (không phải SPEAK/IDLE).
    Trả về tag đầu tiên tìm được (hoặc None).
    """
    matches = TAG_PATTERN.findall(text)
    if not matches:
        return None

    triggered_tag = f"[{matches[0]}]"

    # Chỉ trigger gesture đặc biệt ở đây.
    # [SPEAK] và [IDLE] được quản lý riêng bởi audio timing logic.
    if triggered_tag not in ("[SPEAK]", "[IDLE]"):
        key = LOGIC_TAG_KEY_MAP.get(triggered_tag)
        if key:
            pyautogui.press(key)
            print(f"🎭 Gesture triggered: {triggered_tag} → key '{key}'")

    return triggered_tag


async def send_realtime_audio(session):
    """Đọc từ Mic và gửi lên Gemini"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)

    SILENCE_THRESHOLD = 800
    is_speaking = False

    def get_rms(data):
        count = len(data) // 2
        samples = struct.unpack(f'{count}h', data)
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / count) if count > 0 else 0

    print("🎤 Mic đang mở, hãy nói gì đó...")
    try:
        while True:
            data = await asyncio.to_thread(stream.read, 2048, False)

            rms = get_rms(data)
            if rms > SILENCE_THRESHOLD:
                if not is_speaking:
                    print(f"🔊 Mic đang nhận giọng nói... (RMS={rms:.0f})")
                    is_speaking = True
            else:
                if is_speaking:
                    print(f"🔇 Mic im lặng (RMS={rms:.0f})")
                    is_speaking = False

            await session.send_realtime_input(
                media=types.Blob(data=data, mime_type="audio/pcm")
            )
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Mic error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def receive_and_process(session):
    """Nhận phản hồi, lưu file WAV và điều khiển animation theo logic tags + độ dài audio"""
    full_audio = bytearray()
    # Transcript được tích lũy từ output_transcription (stream theo chunk)
    full_transcript = ""
    # Flag để chắc chắn gesture chỉ trigger 1 lần mỗi lượt
    gesture_triggered = False

    try:
        while True:
            async for message in session.receive():
                server = message.server_content
                if not server:
                    continue

                # --- Thu thập audio data ---
                if server.model_turn:
                    for part in server.model_turn.parts:
                        if part.inline_data:
                            full_audio.extend(part.inline_data.data)

                # --- Thu thập transcript từ output_audio_transcription ---
                # Đây là cách đúng với native audio model (thay vì TEXT modality)
                # Transcript đến theo từng chunk nhỏ song song khi audio stream về
                if server.output_transcription:
                    chunk = server.output_transcription.text or ""
                    if chunk:
                        full_transcript += chunk
                        # Trigger gesture ngay khi parse được tag ở chunk đầu tiên
                        if not gesture_triggered and TAG_PATTERN.search(full_transcript):
                            detected_tag = parse_and_trigger_gesture(full_transcript)
                            if detected_tag:
                                gesture_triggered = True
                                print(f"🏷️  Logic tag: {detected_tag}")

                # --- Kết thúc 1 lượt nói ---
                if server.turn_complete:
                    MIN_AUDIO_BYTES = 24000
                    if len(full_audio) >= MIN_AUDIO_BYTES:
                        # 1. Ghi đè file WAV
                        with wave.open(WAV_PATH, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(24000)
                            wf.writeframes(bytes(full_audio))

                        # 2. Tính độ dài audio (giây)
                        duration = len(full_audio) / 48000.0
                        print(f"✅ Đã lưu file: {WAV_PATH} | ⏱️ Độ dài: {duration:.2f}s")
                        print(f"📝 Transcript: {full_transcript.strip()}")

                        if not gesture_triggered:
                            print("🏷️  Không tìm thấy logic tag → dùng SPEAK mặc định")

                        # 3. Trigger Speaking Animation (phím M) — kích hoạt lip-sync Audio2Face
                        print("🎬 Start Talking Animation (M)")
                        pyautogui.press('m')

                        # 4. Trả về Idle sau khi audio kết thúc
                        async def stop_anim_after_delay(delay):
                            await asyncio.sleep(delay)
                            pyautogui.press('n')
                            print("🛑 Back to Idle Animation (N)")

                        asyncio.create_task(stop_anim_after_delay(duration))

                    elif full_audio:
                        print(f"⚠️ Bỏ qua audio quá ngắn ({len(full_audio)} bytes)")

                    # Reset cho lượt tiếp theo
                    full_audio = bytearray()
                    full_transcript = ""
                    gesture_triggered = False

    except Exception as e:
        print(f"Receive error: {e}")


KNOWLEDGE_PATH = r"D:\FPT Edu\SP26\SWD392\TTS_Middleware\Python_TTS\knowledge_base.md"


def load_knowledge_base(path: str) -> str:
    """Đọc file .md và trả về nội dung, bỏ qua nếu không tìm thấy file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"📚 Đã load knowledge base: {path} ({len(content)} ký tự)")
        return content
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy knowledge base: {path}, dùng hướng dẫn mặc định.")
        return ""


# ---------------------------------------------------------------------------
# LOGIC TAG INSTRUCTIONS — Nhúng vào system prompt để AI biết cách dùng tags
# ---------------------------------------------------------------------------
LOGIC_TAG_INSTRUCTIONS = """
## Hướng dẫn sử dụng Logic Tags (BẮT BUỘC)
Ở ĐẦU mỗi câu trả lời, bạn PHẢI chèn đúng 1 logic tag phù hợp nhất với nội dung.
Không được chèn tag ở giữa hay cuối câu. Chỉ dùng 1 tag duy nhất mỗi lượt.

Danh sách tag và khi nào dùng:
- [WAVE]    → Dùng khi chào hỏi, tạm biệt, giới thiệu bản thân lần đầu.
- [NOD]     → Dùng khi đồng ý, xác nhận, trả lời "có / đúng / chính xác".
- [POINT]   → Dùng khi chỉ dẫn vị trí, hướng dẫn đường đi, giới thiệu địa điểm.
- [THINK]   → Dùng khi cần cân nhắc, câu hỏi phức tạp, hoặc trả lời về kế hoạch/lịch trình.
- [SHRUG]   → Dùng khi không biết, không chắc, hoặc nói "Tôi chưa có thông tin".
- [EXCITED] → Dùng khi thông báo tin vui, sự kiện đặc biệt, hoặc câu hỏi về thành tích.
- [SPEAK]   → Dùng cho tất cả các trường hợp còn lại (trả lời thông thường).

Ví dụ đúng:
- "[WAVE] Xin chào! Tôi là Sumadi, rất vui được gặp bạn!"
- "[POINT] Phòng hội thảo nằm ở tầng 3, rẽ trái sau thang máy."
- "[SHRUG] Tôi chưa có thông tin về vấn đề này, bạn có thể hỏi ban tổ chức nhé."
- "[NOD] Đúng rồi, sự kiện bắt đầu lúc 9 giờ sáng."
"""


async def main():
    base_instruction = "Bạn là trợ lý Metahuman. Trả lời thân thiện, ngắn gọn bằng tiếng Việt."
    knowledge = load_knowledge_base(KNOWLEDGE_PATH)
    system_instruction = (
        f"{base_instruction}\n\n---\n\n{knowledge}\n\n---\n\n{LOGIC_TAG_INSTRUCTIONS}"
        if knowledge
        else f"{base_instruction}\n\n---\n\n{LOGIC_TAG_INSTRUCTIONS}"
    )

    config = {
        "system_instruction": system_instruction,
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Fenrir"}}
        },
        # Native audio model CHỈ hỗ trợ AUDIO — không dùng được ["AUDIO", "TEXT"]
        "response_modalities": ["AUDIO"],
        # Dùng output_audio_transcription để nhận transcript song song với audio
        # Đây là cách đúng để parse logic tags với native audio model
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                "silence_duration_ms": 800,
            },
        },
    }

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        print("🚀 Middleware Connected!")
        await asyncio.gather(
            send_realtime_audio(session),
            receive_and_process(session)
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng Middleware.")
    except Exception as e:
        print(f"\n❌ Lỗi hệ thống: {e}")