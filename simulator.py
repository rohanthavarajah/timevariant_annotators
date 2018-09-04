import numpy as np
import pandas as pd
import dill
import os

def path(path_type, Nj, p1=0.9, p2=0.65, switch=0.5):
    if path_type=="path1":
        '''
        linear
        p1 = start
        p2 = end
        '''
        prob_fn = lambda t: t/Nj*(p2-p1)+p1

    if path_type=="path2":
        '''
        logit
        p1 = start
        p2 = end
        '''
        switch = (switch - 0.5)/2
        prob_fn = lambda t: (p1-p2)/(1+np.exp(-5+10/Nj*(t-switch*Nj)))+p2

    if path_type=="path3":
        '''
        trampoline
        p1 = level
        p2 = bounce
        '''
        def prob_fn(t):
            if t>(Nj/3) and t<(Nj*2/3):
                return (p1-p2)*36/(Nj)**2*(t-Nj/2)**2+p2
            if t>=(Nj*2/3):
                return p1
            if t<=(Nj/3):
                return p1

    if path_type=="path4":
        '''
        piecewise
        '''
        def prob_fn(t):
            if t>switch*Nj:
                return p2
            else:
                return p1
            
    if path_type=="path5":
        '''
        static
        '''
        def prob_fn(t):
            return p1
    return prob_fn

def gen_simulated_data(input_dir, N=500, J=5, K=4, anno_per_post=[5], path_cfg="evolving", shuffle=False):
    beta_true = np.ones(K) # beta_true = np.exp(np.random.normal(0,0.25,K)) 
    rho_true = np.random.dirichlet(alpha=[10,10], size=K)[:,1]
    annotator_subset = anno_per_post

    post_hash = np.arange(N)
    # brand hash
    cuts_props = np.random.dirichlet(np.ones(K)*1000)
    cuts = [round(N*p) for p in cuts_props]
    cuts[-1] = N-np.sum(cuts[:-1]) 
    cuts = np.array(cuts, dtype=int)
    brand_hash = []
    for k in range(K):
        brand_hash = brand_hash + [k]*cuts[k]
        
    # selection mechanism
    correct_response = []
    for k in brand_hash:
        correct_response.append(np.random.multinomial(n=1, pvals=[1-rho_true[k], rho_true[k]])[1])
        
    # worker allocation
    annotators_assigned_to_post = np.random.choice(annotator_subset, size=N, replace=True)
    allocation = [np.random.choice(np.arange(J), size=j, replace=False) for j in annotators_assigned_to_post]
    
    # dataframe
    sim_df_input = []
    for i in range(N):
        for worker_hash in allocation[i]:
            row = {}
            row["post_hash"] = post_hash[i]
            row["brand_id"] = brand_hash[i]
            row["allocation"] = allocation[i]
            row["z"] = correct_response[i] 
            row["rho"] = rho_true[brand_hash[i]]
            row["worker_id"] = "w"+str(worker_hash)
            row["beta"] = beta_true[brand_hash[i]]
            sim_df_input.append(row)
    sim_df = pd.DataFrame(sim_df_input)

    # generate time stamp
    if shuffle==True:
        # shuffle
        sim_df = sim_df.sample(frac=1).reset_index(drop=True)
        sim_df["randomizer"] = sim_df.index
        # sim_df = sim_df.sort_values("worker_id")
        sim_df["t_id"] = sim_df.groupby('worker_id')["randomizer"].rank(ascending=True).astype(int)
        sim_df["t_id"] = sim_df["t_id"]-1
    else:
        sim_df["t_id"] = sim_df["post_hash"]
    
    # simulate paths
    sim_df["Nj"] = sim_df["worker_id"].value_counts()[sim_df["worker_id"]].values
    path_df = pd.DataFrame(sim_df["worker_id"].value_counts()).rename(columns={"worker_id":"Nj"})
    path_col = []
    
    if path_cfg=="static":
        for Nj in path_df["Nj"].values:
            p1 = np.random.uniform(0,2)
            quadrants = [[0.5,1],[1.5,2]]
            p2 = np.random.uniform(*quadrants[1-round(p1/2)])
            path_params = {
            "path_type":np.random.choice(["path5"]), 
            "p1":p1,
            "p2":p2,
            "switch":np.random.uniform(0.1,0.9)
            }
            path_col.append(path(Nj=Nj, **path_params)) 
    elif path_cfg=="easy":
        for Nj in path_df["Nj"].values:
            quadrants = [[-2,-0.5],[0.5,2]]
            p1 = np.random.uniform(*quadrants[np.random.choice([0,1], p=[0.5,0.5])])
            p2 = np.random.uniform(*quadrants[1-round((p1+2)/4)])
            path_params = {
            "path_type":np.random.choice(["path4"]), 
            "p1":p1,
            "p2":p2,
            "switch":0.5,
            }
            path_col.append(path(Nj=Nj, **path_params))
    elif path_cfg=="evolving":
        for ind, Nj in enumerate(path_df["Nj"].values):
            quadrants = [[0,0.25],[2,2.5]] 
            start = np.random.choice([0,1])
            p1 = np.random.uniform(*quadrants[start]) # np.random.choice([0,1])
            p2 = np.random.uniform(*quadrants[1-start])
            path_params = {
            "path_type":["path1","path2","path3","path4","path2"][ind], 
            "p1":p1,
            "p2":p2,
            "switch":0.5,
            }
            path_col.append(path(Nj=Nj, **path_params))
            
    path_df["prob_fn"] = path_col
    path_df = path_df.reset_index().rename(columns={"index":"worker_id"})
    
    # append theta col
    sim_df = sim_df.merge(path_df, on=["worker_id"])
    theta_col = []
    for beta, t_id, prob_fn in sim_df[["beta","t_id","prob_fn"]].values:
        alpha = prob_fn(t_id)
        theta_col.append(1/(1+np.exp(-alpha*beta)))
    sim_df["theta"] = theta_col
    
    # reported response
    sim_df["coin-flip"] = np.random.uniform(size=sim_df.shape[0])
    reported_response = []
    for z, theta, coin in sim_df[["z","theta","coin-flip"]].values:
        if coin<=theta:
            reported_response.append(z)
        else:
            reported_response.append(1-z)
    sim_df["answer"] = np.array(reported_response, dtype=np.int64)
    
    # clean up
    sim_df.drop(['prob_fn', 'allocation', 'coin-flip', 'Nj_x', 'Nj_y'], axis=1, inplace=True)
    sim_df = sim_df.rename(columns={
        "beta":"hidden_beta",
        "rho":"hidden_rho",
        "z":"hidden_z",
        "theta":"hidden_theta"})
    sim_df = sim_df[["brand_id", "worker_id", "post_hash", "answer", "t_id", "hidden_beta", "hidden_rho", "hidden_z", "hidden_theta"]]
    
    # export csvs
    for brand_id in sim_df["brand_id"].unique():
        df = sim_df.loc[sim_df["brand_id"]==brand_id,:]
        filename = "simul_brand_" + str(brand_id) + ".csv"
        df.to_csv(os.path.join(input_dir, filename), index=False)
    
    filename = os.path.join(input_dir, "path_df.dill")
    serialized = dill.dumps(path_df.to_dict())
    with open(filename,'wb') as file_object:
        file_object.write(serialized)

def prep_sim(sim_csvs):
    # import and concatenate
    df_brands = [pd.read_csv(csv) for csv in sim_csvs]
    sim_df = pd.concat(df_brands)
    sim_df = sim_df.reset_index(drop=True)
    
    # generate contiguous ids
    ii_transformer = sim_df["post_hash"].unique()
    ii_transformer = pd.Series(np.arange(len(ii_transformer)), ii_transformer)       
    sim_df["i_uniq"] = ii_transformer[sim_df["post_hash"]].values

    jj_transformer = sim_df["worker_id"].unique()
    jj_transformer = pd.Series(np.arange(len(jj_transformer)), jj_transformer)       
    sim_df["j_uniq"] = jj_transformer[sim_df["worker_id"]].values

    kk_transformer = sim_df["brand_id"].unique()
    kk_transformer = pd.Series(np.arange(len(kk_transformer)), kk_transformer)       
    sim_df["k_uniq"] = kk_transformer[sim_df["brand_id"]].values

    sim_df["flag_hashtag"] = 0
    
    # oracle setting
    sim_df["z_obs"] = -999 
    N = len(ii_transformer)
    observed_ids = np.random.choice(np.arange(N), 30) # 30 posts are observed
    sim_df.loc[sim_df["post_hash"].isin(observed_ids),"z_obs"] = sim_df["hidden_z"]
    sim_df["r"] = sim_df["answer"].values*1
    return sim_df, ii_transformer, jj_transformer, kk_transformer
    
if __name__=="__main__":
	np.random.seed(2139)
	gen_simulated_data("input", N=500, J=5, K=4, anno_per_post=[5], path_cfg="evolving", shuffle=False)