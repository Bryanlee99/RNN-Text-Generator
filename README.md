# RNN-Text-Generator
TensorFlow RNN to generate text based on Shakespeare

I refactored the code to be compatible with TensorFlow v1.0+ using the tf.contrib library, among other code modifications. I selected the ‘GRU’ cell type over the LSTM, custom, and basic cell types as the ‘GRU’ showed the lowest average training loss as per the article. The ‘num_steps’ parameter was reduced from 80 to 40 because the larger ‘num_steps’ would capture more word context at the cost of immediate spelling accuracy. Considering the small training set, context would likely be obscured regardless, so I decided to make the trade-off. The RNN was trained on ‘tinyshakespeare.txt’ and a 100,000 character output was generated.  

A text-similarity algorithm was also made to quantify the validity of words compared to those in the dictionary. The algorithm compares the character difference between each word in the generated text and the most likely intended word in the dictionary, and computes an average percent character error to word length for the text.  

The final 100,000 character text had ~12% word accuracy. Results are displayed in the accompanying text files. 
