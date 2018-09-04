1. Paste the "confidential" folder into the project directory. Run the bash file **organize_dir.sh** to organize the confidential folder. `source organize_dir.sh`.


2. Setup the project environment using the yml file. `conda env create -f tv_annos.yml`.


3. To produce the main result of the paper run **tribe_010218.py**. **tribe_010218.py** takes one command line argument; a string specifying a configuration of model design decisions. In lines 460-472 these configurations are detailed:

>   `cfg_dict["cfg1"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":"out/trace_cfg1.pkl"}` 

>    `cfg_dict["cfg2"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":"out/trace_cfg2.pkl"}`

- The time-variant skill model `cfg3` yielded optimal results (84.0% accuracy on posts where the inferred label of the model and majority voting diverged). However, if we assume annotator skill is constant over time, `cfg11` yields a cheaper static model alternative with comparable accuracy (83.3%).


- If the trajectory of annotator skill is of interest do `python tribe_010218.py cfg3`. If resources are limited do `python tribe_010218.py cfg11`. 


4. For a demo (and explanation) of the model on simulated data see the notebook **demo_on_simulated_data.ipynb**.


5. For an example of how to munge the pymc3 trace to get at outcomes of interest see **results.ipynb**.


6. To generalize the code to new brands edit lines 479-493 of **tribe_010218.py** , which creates a list of paths to brand csvs.

>         tribe_csvs = [
            os.path.join("input","rohan_11977_1513051779.csv"),
            os.path.join("input","rohan_13584_1513051779.csv"),
            os.path.join("input","rohan_14937_1513051907.csv"),
            ...
            ]
    
    
- In addition update **brand_lkup.py** with the new brand id, brand name and hashtags.


7. Note that multiprocessing nets very little speedup with MCMC due to the sequential nature of sampling. Do NOT enable GPU utilization (pymc3's GPU utilization is still in development and slows everything down).

