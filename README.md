# DeepRM_TC
 Rating prediction models

**Cognitive processes-driven model design: A deep learning recommendation model with textual review and context**

## Usage

**Requirements**

- Python >= 3.6
- Pytorch >= 1.0
- fire: commend line parameters (in `config/config.py`)
- numpy, gensim etc.


**Use the code**

- Preprocessing the datasets via `pro_data/data_pro.py`, then some `npy` files will be generated in `dataset/`, including train, val, and test datset.
    ```
    1. First download the dataset form 'http://BiDMA.quickconnect.to/d/s/t6DflJ5Rb28PX7Emwl9QZPLWsxJ287oh/-st6CO2GQxlrVOL8LHMxOT-AQ005pecG-OLuARl_2WQo' and placing it under the "given_data" folder
    2. Run the data_pro_Cross_validation.py
    ```
    
- Train the model and test the model using the saved pth file in results/chechpoints
    ```
    run the methods.py
    ```

## Citation
If you use the code, please cite:
```

```

## Reference

```
https://github.com/ShomyLiu/Neu-Review-Rec
```



