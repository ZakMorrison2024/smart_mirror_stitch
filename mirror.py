import pygame
import time

# Initialize Pygame for Audio Playback
pygame.mixer.init()

# Directory for Stitch's Audio Clips
AUDIO_DIR = "stitch_audio"

# Function to Play Stitch's Audio Clips
def play_stitch_audio(filename):
    audio_path = f"{AUDIO_DIR}/{filename}"
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # Wait until playback finishes

# Function to Handle Commands with Pre-Recorded Audio
def handle_command_with_audio(command):
    if "weather" in command:
        play_stitch_audio("weather.mp3")
    elif "time" in command:
        current_time = time.strftime("%I:%M %p")
        print(f"The time is {current_time}.")
        play_stitch_audio("time.mp3")  # Pre-recorded audio can say something generic
    elif "hello" in command:
        play_stitch_audio("hello.mp3")
    else:
        play_stitch_audio("unknown.mp3")  # Fallback for unrecognized commands
