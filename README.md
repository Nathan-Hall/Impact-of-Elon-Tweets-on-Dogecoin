#Impact-of-Elon-Tweets-on-Dogecoin

This was a project made by Nathan Hall for ECMT3150 at USYD.

##### Provided under MIT License by Nathan Hall.
*Note: this library may be subtly broken or buggy. The code is released under
the MIT License – please take the following message to heart:*
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Explanation
- Retrieves one-minute data from exchange api's within a window of time around Elon Musks's tweets about Dogecoin. 
- Calculates abnormal return, cumulative abnormal return, average abnormal return and cumulative average abnormal reuturn of the data. Also runs a Wilcoxon signed rank test on each minute since the tweet.
- Data is taken from Binance API, unless there is missing data due to server maintenance, in which case it is taken from FTX.
- All inputs are changeable in config.json.
- Before running you have to install all packages, which you can do with:
'''python
$ pip install -r requirements.txt
'''

P.S: I am in my final year, so please check out my LinkedIn and hire me :)