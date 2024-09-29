# Music recommendation system

Music recommendations are based on the two scores: user scores and internal score. While the user score is static, the internal score is dependent on the act of listening. The initial internal score is always 0, it gets higher if the user listens to the whole song and gets lower if the user skips it. 


# Calculating probabilies for song selection
## User Ratings and Missing Data
The starting point is user ratings. If a song has a rating, it uses that; if not, it uses a fallback mechanism.  

- **User rating**: $ R_u $
- **Model rating**: $ R_m $
- **Mean user rating**: $ R_{\text{mean}} $

For each song, the system checks whether a user or model rating exists. If neither is present, the system assumes the rating is the mean value of all user ratings.  

$$
R = \begin{cases} 
R_u & \text{if } R_u \text{ exists} \\
R_m & \text{if } R_u \text{ does not exist and } R_m \text{ exists} \\
R_{\text{mean}} & \text{if neither } R_u \text{ nor } R_m \text{ exists}
\end{cases}
$$

This ensures every song gets a score.

## Adjusting and Normalizing Scores
Once the ratings are gathered, they need to be adjusted for fairness. The system normalizes them and adds some weight to higher-rated songs. The normalization process is:

**Ensure no score is zero** by adding a small constant (0.1) to the score:  

$ R' = \max(0.1, R) $  

**Amplify high ratings** by normalizing the scores and then squaring them to give higher-rated songs more weight: 

$ R_{\text{adjusted}} = \left( \frac{R'}{10} \right)^2 $  

This squaring makes songs with higher ratings much more likely to be chosen, while ensuring lower-rated songs still have a small chance.

## Skip and Play History
The next adjustment comes from considering how often a song has been played versus skipped:

- **Full plays**: $ P_f $
- **Skips**: $ P_s $

The **skip score** is calculated using the difference between full plays and skips, normalized by an empirically chosen factor of 5.  

$ S_{\text{skip}} = \sigma \left( \frac{5 + P_f - P_s}{5} \right) $  

where σ(x) is the **sigmoid function**:  

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

This ensures that the skip score stays in a meaningful range between 0 and 1.

## Time Since Last Played
To promote variety, the system boosts the chances for songs that haven’t been played recently. The system calculates a **last played score** based on how recently the song was played compared to other songs.  

- **Index of song** in the list sorted by last played: $ I_{\text{last}} $
- **Total number of songs**: $ N $

The last played score is calculated as:  

$ S_{\text{time}} = \frac{I_{\text{last}}}{N} $

Songs that haven’t been played in a long time get a higher score, while recently played songs get a lower score.

## Final Score Calculation
Now that we have all the factors:  

- Adjusted rating: $ R_{\text{adjusted}} $
- Skip score: $ S_{\text{skip}} $
- Last played score: $ S_{\text{time}} $

The final score for each song is a product of these three components:  

$ S_{\text{final}} = R_{\text{adjusted}} \times S_{\text{skip}} \times S_{\text{time}} $

This final score \( S_{\text{final}} \) represents how likely a song is to be selected.  

## Calculating Probabilities
Once the final scores are calculated for all songs, they are converted into probabilities. The probability of a song $ i $ being selected is its score divided by the sum of all scores:  

$ P_i = \frac{S_{\text{final}, i}}{\sum_{j=1}^{N} S_{\text{final}, j}} $

This ensures that the probabilities sum to 1, and each song’s chance of being selected is proportional to its score.  

## Random Selection
Finally, the system uses these probabilities to randomly select a song. A song with a higher probability $ P_i $ is more likely to be chosen, but even songs with lower probabilities still have a chance based on their final score.

## Example Calculation
Let’s say we have three songs with the following details:  


| Song | User Rating | Full Plays | Skips | Last Played (days ago) |
|------|-------------|------------|-------|------------------------|
| A    | 8           | 10         | 2     | 7                      |
| B    | 6           | 5          | 3     | 2                      |
| C    | None        | 2          | 5     | 30                     |

**Adjusted Ratings**:  

- Song A: $ R_{\text{adjusted}} = \left( \frac{8}{10} \right)^2 = 0.64 $
- Song B: $ R_{\text{adjusted}} = \left( \frac{6}{10} \right)^2 = 0.36 $
- Song C uses the mean rating (say 7): $ R_{\text{adjusted}} = \left( \frac{7}{10} \right)^2 = 0.49 $

**Skip Scores**:  

- Song A: $ S_{\text{skip}} = \sigma \left( \frac{5 + 10 - 2}{5} \right) = \sigma(2.6) \approx 0.93 $
- Song B: $ S_{\text{skip}} = \sigma \left( \frac{5 + 5 - 3}{5} \right) = \sigma(1.4) \approx 0.80 $
- Song C: $ S_{\text{skip}} = \sigma \left( \frac{5 + 2 - 5}{5} \right) = \sigma(0.4) \approx 0.60 $

**Last Played Scores** (let’s assume there are 3 songs total):  

- Song A (played 7 days ago): $ S_{\text{time}} = \frac{2}{3} \approx 0.67 $
- Song B (played 2 days ago): $ S_{\text{time}} = \frac{1}{3} \approx 0.33 $
- Song C (played 30 days ago): $ S_{\text{time}} = \frac{3}{3} = 1.0 $

**Final Scores**:  

- Song A: $ S_{\text{final}} = 0.64 \times 0.93 \times 0.67 \approx 0.40 $
- Song B: $ S_{\text{final}} = 0.36 \times 0.80 \times 0.33 \approx 0.095 $
- Song C: $ S_{\text{final}} = 0.49 \times 0.60 \times 1.0 \approx 0.29 $

**Probabilities**:  

$ P_A = \frac{0.40}{0.40 + 0.095 + 0.29} \approx 0.50 $  
$ P_B = \frac{0.095}{0.40 + 0.095 + 0.29} \approx 0.12 $  
$ P_C = \frac{0.29}{0.40 + 0.095 + 0.29} \approx 0.36 $

In this case, Song A has a 50% chance of being selected, Song B has a 12% chance, and Song C has a 36% chance.

# Future plans
In future I would also like to add more different options of building a current playlist. One of the ideas is a “chain mode” that takes selected by the user song as a seed and finds the most similar song, based on the embeddings, then finds the most similar to a next song and so on.

In case there would be a CLIP-like model for audio in the future, I would also like to implement prompt-based playlist generation. One could simply describe the mood of the music they would like to listen and the proper playlist be generated on the fly from this description.
