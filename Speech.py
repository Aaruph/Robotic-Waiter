import pyttsx3

def announce_order_arrival():
    text = "Your order has arrived. Please pick up."

    engine = pyttsx3.init()

    # Get the available voices
    voices = engine.getProperty('voices')

    # Select a male voice
    for voice in voices:
        if 'male' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.say(text)
    engine.runAndWait()
