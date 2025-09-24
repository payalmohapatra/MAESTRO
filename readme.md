# **Minimal Code for Reproduction for "MAESTRO : Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series"**  | Paper ID - 18915

## **Installation**  
To recreate the exact Conda environment, run:  
```bash
conda env create -f environment.yml -n new_env_name
```

## **Running the Code**  
Follow these steps to run the experiment:  

1. **Sample Dataset**  
   - We have provided the processed WESAD dataset here - ./data.
   - Other processed datasets will be made public after the review cycle. 

2. **Scripts for reproducing the primary results in Table 2.**  
   ```bash
   python main_wesad_maestro.py
   python main_dsads_maestro.py
   python main_mimic_maestro.py
   python main_daliahar_maestro.py
   ```

  

## **Project Structure**  
📂 `utils/`  
&nbsp;&nbsp;&nbsp;&nbsp;📄 Contains dataset configurations (dataset_cfg.py) and basic helper functions (helper_function.py).

📂 `models/`  
&nbsp;&nbsp;&nbsp;&nbsp;📄 Contains the main building blocks of MAESTRO.  
