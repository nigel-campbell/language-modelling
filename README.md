# Neural Language Models

A repo for exploring neural language models utilizing the corpus of Trump
Tweets.

## Usage

```
python main.py --data <src>
```

## Analysis TODOs

- [ ] Establish and evaluate baseline loss for train, test, val
- [ ] Evaluate loss at various lookahead values for train, test and val


## Framework TODOs

- [x] Implement data ingestion workflow
- [x] Define basic RNN/LSTM architecture and model
- [x] Implement model training
- [x] Validate decreasing loss
- [x] Implement text generation
- [x] Implement new multi-loss optimization
- [x] Integrate CUDA and test on ICEHAMMER
- [x] Implement model persistence
- [x] Implement metric generation
- [ ] Implement evaluation on train, test and loss.
