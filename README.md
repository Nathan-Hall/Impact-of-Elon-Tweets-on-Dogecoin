# Impact-of-Elon-Tweets-on-Dogecoin

This was a project made by Nathan Hall for ECMT3150 at USYD.

##### Provided under MIT License by Nathan Hall.

## Explanation
- Retrieves one-minute data from Binance API within a window of time around Elon Musks's tweets about Dogecoin (Feeds into FTX if there is missing data). 
- Calculates abnormal return, cumulative abnormal return, average abnormal return and cumulative average abnormal reuturn of the data. Also runs a Wilcoxon signed rank test on each minute since the tweet.
- Also takes the market cap of each day of the event from coingecko API.
- Runs multiple cross-section regression on every minute CAR from the tweet to 60 minutes after with the model:

```markdown
CAR = Tweet_Count + Market_Cap + Market_Cap*Tweet_Count + Link + Video + Picture + Recognisability + Month + Past_Hout_Ret + Past_Hour_Vol
```
- Plots all coefficients and pvalues using Matplotlib over time, so it's possible to view changing marginal impact as a function of time (see Graphs).
- Also runs multiple ts regression to check for changes in autocorrelation structure after an event:
```markdown
Return = Event_Dummy + Return_1 + Event_Dummy*Return1
```
- Outputs an excel file that holds all data and calculation results.
- All inputs are changeable in config.json.
- More sophisticated econometric analysis in the R-studio file but very messy as of now (oops)
- Before running you have to install all packages, which you can do with:

```python
$ pip install -r requirements.txt
```

P.S: I am self-taught coder so code probs not the best
