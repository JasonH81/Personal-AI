# Personal-AI

This is a python project that utilizes ChatGPT 3.5 and ElevenLabs along with a neural network model to train on a specific voice to learn what someone sounds like. If the model learns what specific features makes up a voice and recognizes it, then it will run a function to put the said speech through ChatGPT. The following text will be put into Elevenlabs and read out via a voice. If the model does not recognize the voice then they will be denied access to the program.

# To-Do

- Separate the voice-training process and voice-detection process into two different projects so that the program can be run without retraining the neural network model every time
- Allow the program to run in the background on your computer
- Force the program to only respond to when a certain word is said, kind of like how modern ai assistants like alexa or siri work.
