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
#   [WAVE]  → phím 1  — Vẫy tay TRƯỚC, sau đó trigger M để nói
#   [SPEAK] → chỉ trigger phím M (nói bình thường)
#   Phím M  — Start Talking Animation (lip-sync Audio2Face)
#   Phím N  — Back to Idle
# ---------------------------------------------------------------------------
WAVE_TAG_PATTERN = re.compile(r'\[WAVE\]')

# Flag toàn cục: True khi bot đang phát audio → tạm dừng thu mic
bot_is_speaking = False


async def send_realtime_audio(session):
    """Đọc từ Mic và gửi lên Gemini — tạm dừng khi bot đang nói"""
    global bot_is_speaking

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
            # Nếu bot đang nói → đọc mic nhưng KHÔNG gửi lên Gemini
            data = await asyncio.to_thread(stream.read, 2048, False)

            if bot_is_speaking:
                # Drain buffer mic nhưng bỏ qua, không gửi
                await asyncio.sleep(0.01)
                continue

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
    """Nhận phản hồi, lưu file WAV và điều khiển animation"""
    global bot_is_speaking

    full_audio = bytearray()
    full_transcript = ""
    is_wave = False

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

                # --- Thu thập transcript, detect [WAVE] sớm ---
                if server.output_transcription:
                    chunk = server.output_transcription.text or ""
                    if chunk:
                        full_transcript += chunk
                        if not is_wave and WAVE_TAG_PATTERN.search(full_transcript):
                            is_wave = True
                            print("🏷️  Logic tag: [WAVE] → Trigger wave gesture (key '1')")
                            pyautogui.press('1')

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

                        # 2. Tính độ dài audio
                        duration = len(full_audio) / 48000.0
                        print(f"✅ Đã lưu file: {WAV_PATH} | ⏱️ Độ dài: {duration:.2f}s")
                        print(f"📝 Transcript: {full_transcript.strip()}")

                        # 3. Khóa mic trước khi trigger animation
                        bot_is_speaking = True
                        print("🔒 Mic tạm dừng (bot đang nói)")

                        # 4. Trigger talking animation + lip-sync
                        print("🎬 Start Talking Animation (M)")
                        pyautogui.press('m')

                        # 5. Mở mic lại + trả về Idle sau khi audio kết thúc
                        async def stop_anim_after_delay(delay):
                            global bot_is_speaking
                            await asyncio.sleep(delay)
                            pyautogui.press('n')
                            print("🛑 Back to Idle Animation (N)")
                            bot_is_speaking = False
                            print("🔓 Mic mở lại (bot đã nói xong)")

                        asyncio.create_task(stop_anim_after_delay(duration))

                    elif full_audio:
                        print(f"⚠️ Bỏ qua audio quá ngắn ({len(full_audio)} bytes)")

                    # Reset cho lượt tiếp theo
                    full_audio = bytearray()
                    full_transcript = ""
                    is_wave = False

    except Exception as e:
        print(f"Receive error: {e}")


KNOWLEDGE_PATH = r"D:\FPT Edu\SP26\SWD392\TTS_Middleware\Python_TTS\knowledge_base.md"


def load_knowledge_base(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"📚 Đã load knowledge base: {path} ({len(content)} ký tự)")
        return content
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy knowledge base: {path}, dùng hướng dẫn mặc định.")
        return ""


# ---------------------------------------------------------------------------
# LOGIC TAG INSTRUCTIONS
# ---------------------------------------------------------------------------
LOGIC_TAG_INSTRUCTIONS = """
## Hướng dẫn sử dụng Logic Tags (BẮT BUỘC - TUYỆT ĐỐI TUÂN THỦ)
Ở ĐẦU mỗi câu trả lời, bạn PHẢI chèn đúng 1 trong 2 logic tag sau:

- [WAVE]  → BẮT BUỘC dùng khi: người dùng chào ("xin chào", "hello", "hi", "chào"), 
            khi tạm biệt ("bye", "tạm biệt", "hẹn gặp lại"), 
            khi bạn tự giới thiệu bản thân lần đầu,
            hoặc khi người dùng YÊU CẦU bạn vẫy tay.
            Dù đã chào nhiều lần trước, MỖI LẦN người dùng nói lời chào THÌ VẪN PHẢI dùng [WAVE].
- [SPEAK] → Dùng cho tất cả các trường hợp còn lại.

Chỉ dùng 1 tag duy nhất, đặt ở đầu câu. KHÔNG đặt tag ở giữa hay cuối câu.

Ví dụ đúng:
- Người dùng nói "xin chào"  → "[WAVE] Xin chào! Tôi là Sumadi..."
- Người dùng nói "hello"     → "[WAVE] Hello bạn! Rất vui được gặp bạn!"
- Người dùng nói "vẫy tay đi" → "[WAVE] Đây này! Tôi vẫy tay chào bạn nè!"
- Người dùng hỏi về trường   → "[SPEAK] FPT University có 5 cơ sở toàn quốc."
- Người dùng nói "tạm biệt"  → "[WAVE] Tạm biệt bạn, hẹn gặp lại!"
"""


async def main():
    base_instruction = "Bạn là trợ lý Metahuman tên Sumadi. Trả lời thân thiện, ngắn gọn bằng tiếng Việt."
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
        "response_modalities": ["AUDIO"],
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