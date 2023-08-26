# Scoring

Given a WAV file, we want to assign it a score measuring the quality of delivery i.e. how dynamic or monotonous the speaker was.

There are many features one can look at and we've sorted out below those that are / aren't relevant.

## Relevant features

- intensity std: accounts for volume variation
- pitch std: accounts for pitch (which is basically the fundamental frequency of the speaker's voice) variation
- harmonicity mean: accounts for the speaker's ability to raise his voice below ambient noise (higher harmonicity = less background noise pollution)

## Irrelevant features

- intensity mean: irrelevant since (apparently) the browser / the video is always rescaled in volume i.e. upon saving a recording of someone shouting / whispering while have roughly the same intensity mean
