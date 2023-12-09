# This page at the draft stage!
# Music recomendation system

Music recommendations are based on the two scores: user scores and internal score. While the user score is static, the internal score is dependent on the act of listening. The initial internal score is always 0, it gets higher if the user listens to the whole song and gets lower if the user skips it. 

I would recommend to keep all ratings on the lower end, as all the music you have locally most likely will be preferred by you, so it will tend to have roughly the same high score. Try to keep 9 only for the best song you have and not use 10 at all until you hear something exceptionally good.

## Song rating estimation with LLM

The main idea is to create one-to-one correspondence in LLM with the sound in a similar way how VQ-VAE works. But if in VQ-VAE all tokes are learned during the training, in our case we already have all the tokens available fixed, therefore we can only use separate trained network to create correspondence between those tokens and audio segments. After this correspondence is crated and the new network is trained we can simply sample few fragments from the song and ask LLM how would it rate this song. Something like this:

```
Here is few sample from the song. Rate this song on the scale from 0 to 10.

[music]{first song sample tokes}[/music]
[music]{second song sample tokes}[/music]
[music]{third song sample tokes}[/music]

Rating:
```

Then we train LLM with q-LORA method by the example of ratings we already have in the music library that user gave,

## Radio mode

LLM used as an AI DJ. The big goal is to create a virtual AI DJ that could understood by a few words what kind of music you want to listen right now and play it accordigly by your current and long term feedback.