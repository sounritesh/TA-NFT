# TA-NFT
### Steps to train the model:
1. Download and extract the data zip file.
3. Encode the preprocessed tweets using BERTweet and save the encodings in a ".npy" file.
4. Keep all the extracted files and encodings in the same folder.
5. Install all dependencies using:
  ```
  pip install -r requirements.txt
  ```
6. Train the model using the commands:
  - For Daily Average Price Prediction:
    ```
    python run.py --data_dir $PATH_TO_DATA_DIR --model rtlstm_hawkes --output_dir $PATH_TO_SAVE_CHECKPOINTS --epochs 20 
    ```
  - For Price Movement Classification:
    ```
    python run.py --data_dir $PATH_TO_DATA_DIR --model rtlstm_hawkes --output_dir $PATH_TO_SAVE_CHECKPOINTS --epochs 20 --classification
    ```
Note: You may add/edit other hyperparamters to checkout different configurations.
