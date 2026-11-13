
# AI Clip Factory — Project Instructions

## Project goal
Build a local web application that automatically processes authorized Twitch VOD recordings and creates vertical YouTube Shorts, Instagram Reels and TikTok videos.

## Hardware
- CPU: Intel Core i5-11400F
- RAM: 16 GB
- GPU: NVIDIA GTX 1050 Ti
- OS: Windows 10/11

## Main requirements
- Use free and open-source tools whenever possible.
- Do not require paid APIs.
- Process long recordings (2–5 hours).
- Main content language: English.
- Generate vertical 9:16 videos.
- Detect interesting moments automatically.
- Generate animated subtitles.
- Add dynamic zooms, transitions and effects.
- Generate titles, descriptions and hashtags.
- Provide a web interface.
- Process videos locally whenever practical.

## Technology stack
Backend:
- Python 3.12
- FastAPI
- SQLite initially
- faster-whisper
- Silero VAD
- FFmpeg

Frontend:
- React
- JavaScript
- Vite
- Tailwind CSS

AI:
- Free coding models through OpenCode
- Ollama for lightweight local tasks
- Whisper for speech recognition
- Rule-based and AI-assisted highlight detection

## Architecture rules
- Keep modules independent.
- Use clear interfaces between modules.
- Do not put all application logic into main.py.
- Separate API routes, services, models and configuration.
- Use background jobs for long video processing.
- Store processing status and errors.
- Make processing resumable where practical.
- Avoid loading entire videos into RAM.
- Use FFmpeg for video and audio operations.

## Development rules
- Inspect existing files before editing.
- Preserve working functionality.
- Do not rewrite unrelated files.
- Do not create duplicate modules.
- Use type hints and meaningful names.
- Handle exceptions and log errors.
- Add tests for important functionality.
- Run tests after modifications.
- Explain changed files and test results.
- Never claim a test passed without running it.

## Safety and permissions
- Never expose API keys or secrets.
- Never commit .env files.
- Do not delete user data or original recordings.
- Do not execute destructive Git commands without approval.
- Do not publish or upload videos automatically without explicit authorization.
- Process only recordings the user is authorized to use.

## Workflow
1. Inspect the current project.
2. Understand the task and affected modules.
3. Implement the smallest complete change.
4. Run relevant tests.
5. Fix errors.
6. Review the final diff.
7. Report what changed and what remains.
