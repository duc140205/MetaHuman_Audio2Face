import asyncio
import wave
import struct
import math
import os
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

async def send_realtime_audio(session):
    """Đọc từ Mic và gửi lên Gemini"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
    
    SILENCE_THRESHOLD = 800   # RMS dưới ngưỡng này = im lặng (điều chỉnh nếu cần)
    is_speaking = False

    def get_rms(data):
        count = len(data) // 2
        samples = struct.unpack(f'{count}h', data)
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / count) if count > 0 else 0

    print("🎤 Mic đang mở, hãy nói gì đó...")
    try:
        while True:
            # Dùng to_thread để không block event loop
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

            # Gửi realtime input theo định dạng SDK mới nhất
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
    """Nhận phản hồi, lưu file WAV và điều khiển animation theo độ dài audio"""
    full_audio = bytearray()
    
    try:
        while True:
            async for message in session.receive():
                if message.server_content and message.server_content.model_turn:
                    parts = message.server_content.model_turn.parts
                    for part in parts:
                        if part.inline_data:
                            full_audio.extend(part.inline_data.data)
                
                if message.server_content and message.server_content.turn_complete:
                    MIN_AUDIO_BYTES = 24000
                    if len(full_audio) >= MIN_AUDIO_BYTES:
                        # 1. Ghi đè file WAV
                        with wave.open(WAV_PATH, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(24000) 
                            wf.writeframes(bytes(full_audio))
                        
                        # 2. Tính toán độ dài audio (giây)
                        # Công thức: Tổng số bytes / (Sample Rate * Bytes per Sample * Channels)
                        # Ở đây: len(full_audio) / (24000 * 2 * 1)
                        duration = len(full_audio) / 48000.0
                        print(f"✅ Đã lưu file: {WAV_PATH} | ⏱️ Độ dài: {duration:.2f}s")
                        
                        # 3. Điều khiển Animation qua phím tắt
                        print("🎬 Start Talking Animation (M)")
                        pyautogui.press('m') 

                        # Chạy task chờ và tắt animation mà không block việc nhận audio tiếp theo
                        async def stop_anim_after_delay(delay):
                            await asyncio.sleep(delay)
                            pyautogui.press('n')
                            print("🛑 Back to Idle Animation (N)")

                        asyncio.create_task(stop_anim_after_delay(duration))

                    elif full_audio:
                        print(f"⚠️ Bỏ qua audio quá ngắn ({len(full_audio)} bytes)")
                    
                    full_audio = bytearray()
                
                try:
                    if getattr(message, "text", None):
                        print(f"🤖 AI: {message.text}")
                except Exception:
                    pass
                
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

async def main():
    base_instruction = "Bạn là trợ lý Metahuman. Trả lời thân thiện, ngắn gọn bằng tiếng Việt."
    knowledge = load_knowledge_base(KNOWLEDGE_PATH)
    system_instruction = f"{base_instruction}\n\n---\n\n{knowledge}" if knowledge else base_instruction

    config = {
        "system_instruction": system_instruction,
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Fenrir"}}
        },
        "response_modalities": ["AUDIO"],
        "realtime_input_config": {
            # Gemini tự phát hiện đầu/cuối câu nói phía server
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