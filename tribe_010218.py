import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"
import theano
import theano.tensor as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pymc3 as pm
from pymc3.backends.base import merge_traces
import sys
import json
import brand_lkup

def prep_tribe(tribe_csvs, brand_lookup, keep_prolific_cut=-1, hashtag_treatment="strong", condense_repeat_votes=True, model_decision_as_worker=True):
    '''
    hashtag_treatment = none 
    hasttag_treatment = weak (weak prior)
    hashtag_treatment = strong (strong prior)
    hashtag_treatment = oracle (set z_obs = 1)
    '''
    # import
    df_brands = [pd.read_csv(csv) for csv in tribe_csvs]
    
    # merge
    tribe_df = pd.concat(df_brands)
    tribe_df = tribe_df.merge(brand_lookup, on="brand_id", indicator=True)
    assert np.mean(tribe_df["_merge"]=="both")==1
    del tribe_df["_merge"]

    # pad with leading and trailing spaces
    tribe_df["text"] = " " + tribe_df["text"] + " "
    # add flag_hashtag
    flag_hashtag_column = []
    for text, hashtag in tribe_df[["text","hashtags"]].values:
        flag_hashtag = 0
        for tag in hashtag:
            if tag.lower() in text.lower():
                flag_hashtag=1
        flag_hashtag_column.append(flag_hashtag)
    tribe_df["flag_hashtag"] = np.array(flag_hashtag_column, dtype=int)
    
    # incorporate model_decision as a worker
    if model_decision_as_worker==True:
        model_df = tribe_df.copy()
        del model_df["answer"]
        model_df = model_df[["brand_id","worker_id","post_hash","flag_hashtag","model_decision"]]
        model_df = model_df.groupby(["brand_id","post_hash","flag_hashtag"]) \
            .mean().reset_index() \
            .rename(columns={"model_decision":"answer"})
        model_df["worker_id"]="model"
        tribe_df = tribe_df.append(model_df)

    tribe_df = tribe_df[["brand_id","worker_id","post_hash","answer","flag_hashtag","date"]]

    # condense
    if condense_repeat_votes==True:
        # generate temp t_id lookup
        tribe_df["date"] = pd.to_datetime(tribe_df.date)
        tribe_df = tribe_df.sort_values(by=["worker_id","date"])
        firstdate_df = tribe_df.copy()
        firstdate_df["t_id_tmp"] = firstdate_df.groupby(['worker_id'])['date'].rank(ascending=True)
        firstdate_df = firstdate_df.groupby(["brand_id","worker_id","post_hash","answer"]).min()[["t_id_tmp"]].reset_index()

        # condense
        rows_before = tribe_df.shape[0]
        tribe_df = tribe_df.groupby(["brand_id","worker_id","post_hash","flag_hashtag"]).mean().reset_index()
        # hard cases
        hard_cases_df = tribe_df.loc[tribe_df["answer"]==0.5,:]
        # easy cases
        tribe_df = tribe_df.loc[tribe_df["answer"]!=0.5,:]
        tribe_df["answer"] = 1*(tribe_df["answer"]>0.5)
        hard_cases_df["answer"] = 1
        tribe_df = pd.concat([tribe_df, hard_cases_df])
        hard_cases_df["answer"] = 0
        tribe_df = pd.concat([tribe_df, hard_cases_df])
        print("rows dropped due to condensing = %i" % (rows_before-tribe_df.shape[0]))

        # recalculate t_id
        tribe_df = tribe_df.merge(firstdate_df, on=["brand_id","worker_id","post_hash","answer"], how='left', indicator=True)
        assert np.mean(tribe_df["_merge"]=="both")==1
        del tribe_df["_merge"]
        tribe_df = tribe_df.sort_values(by=["worker_id","t_id_tmp"])
        tribe_df["t_id"] = tribe_df.groupby(['worker_id'])['t_id_tmp'].rank(ascending=True)
        tribe_df["t_id"] = tribe_df["t_id"]-1
        tribe_df["t_id"] = tribe_df["t_id"].astype(int)
    else:
        tribe_df["date"] = pd.to_datetime(tribe_df.date)
        tribe_df = tribe_df.sort_values(by=["worker_id","date"])
        tribe_df["t_id"] = tribe_df.groupby('worker_id')['date'].rank(ascending=True)
        tribe_df["t_id"] = tribe_df["t_id"]-1
        tribe_df["t_id"] = tribe_df["t_id"].astype(int)

    # deal with brand ambiguity
    tribe_df["post_hash"] = tribe_df["post_hash"].astype(str) + "_" + tribe_df["brand_id"].astype(str)

    # get workers with more than 20 labels
    workload = tribe_df.groupby("worker_id").count().reset_index()
    worker_subset = workload[workload["post_hash"]>keep_prolific_cut].worker_id.values
    rows_before = tribe_df.shape[0]
    tribe_df = tribe_df[tribe_df.worker_id.isin(worker_subset)]
    print("rows dropped due to non prolific worker = %i" % (rows_before-tribe_df.shape[0]))

    tribe_df = tribe_df.sort_values(["post_hash","brand_id","worker_id"])

    # oracle consensus
    oracle_df = tribe_df[tribe_df["worker_id"].apply(lambda x: x[:5]=="TRIBE")] \
        .groupby(["post_hash","brand_id"]) \
        .mean() \
        .reset_index()[["post_hash","brand_id","answer"]] \
        .rename(columns={"answer":"z_obs"})
    rows_before = oracle_df.shape[0]
    oracle_df = oracle_df.loc[oracle_df["z_obs"].isin([0,1]), :]
    print("rows dropped due to no tribe consensus = %i" % (rows_before-oracle_df.shape[0]))

    # set up cross validation 
    # for now use all tribe annotations for validation i.e. none for oracle
    validation_df = oracle_df.copy()
    oracle_df = oracle_df[oracle_df["brand_id"]==99999] # WITH MORE OVERLAP REVISIT THIS
    print("validation dataframe has %i posts" % validation_df.groupby(["post_hash"]).sum().shape[0])
    
    # use tribe as oracle for tribe annotations not used for validation
    tribe_df = tribe_df.merge(right=oracle_df, how='left', on=["post_hash","brand_id"])
    tribe_df["z_obs"] = tribe_df["z_obs"].fillna(-999)
    rows_before = tribe_df.shape[0]
    tribe_df = tribe_df[tribe_df["worker_id"].apply(lambda x: x[:5]!="TRIBE")]
    print("tribe rows dropped = %i" % (rows_before-tribe_df.shape[0]))
    tribe_df["z_obs"] = tribe_df["z_obs"].astype(int)

    if hashtag_treatment=="oracle":
        print("z set to 1 due to presence of hashtag for %i rows" % tribe_df[tribe_df["flag_hashtag"]==1].shape[0])
        print("total rows available = %i" % tribe_df.shape[0])
        tribe_df.loc[tribe_df["flag_hashtag"]==1, "z_obs"] = 1
    
    # generate contiguous ids
    ii_transformer = tribe_df["post_hash"].unique()
    ii_transformer = pd.Series(np.arange(len(ii_transformer)), ii_transformer)       
    tribe_df["i_uniq"] = ii_transformer[tribe_df["post_hash"]].values

    jj_transformer = tribe_df["worker_id"].unique()
    jj_transformer = pd.Series(np.arange(len(jj_transformer)), jj_transformer)       
    tribe_df["j_uniq"] = jj_transformer[tribe_df["worker_id"]].values

    kk_transformer = tribe_df["brand_id"].unique()
    kk_transformer = pd.Series(np.arange(len(kk_transformer)), kk_transformer)       
    tribe_df["k_uniq"] = kk_transformer[tribe_df["brand_id"]].values

    tribe_df["r"] = tribe_df["answer"].values*1
    return tribe_df, validation_df, ii_transformer, jj_transformer, kk_transformer

class mcmc_input_class(object):
    def __init__(self, ii, jj, kk, t_id, ii_lkup, kk_lkup, r_obs, z_init, z_obs, flag_hashtag):
        self.ii = ii
        self.jj = jj
        self.kk = kk
        self.t_id = t_id
        self.ii_lkup = ii_lkup
        self.kk_lkup = kk_lkup
        self.r_obs = r_obs
        self.z_init = z_init
        self.z_obs = z_obs
        self.flag_hashtag = flag_hashtag
        
        self.N = max(self.ii)+1
        self.J = max(self.jj)+1
        self.K = max(self.kk)+1

def get_inputs(sim_df):
    '''
    prepares inputs for mcmc
    post-hash should be contiguous (no gaps between 0 and last query id)
    '''
    z_obs = None
    flag_hashtag = None
    
    # indices
    ii = sim_df["i_uniq"].values
    jj = sim_df["j_uniq"].values
    kk = sim_df["k_uniq"].values
    t_id = sim_df["t_id"].values
    r_obs = sim_df["answer"].values*1
    
    # get lookup between posts and brands
    post_brand_lkup = sim_df.groupby(by=["i_uniq","k_uniq"]) \
            .mean() \
            .reset_index()
    
    ii_lkup = np.array(post_brand_lkup["i_uniq"].values, dtype=int)
    kk_lkup = np.array(post_brand_lkup["k_uniq"].values, dtype=int)
    z_obs = np.array(post_brand_lkup["z_obs"].values, dtype=int)
    assert np.sum((1-post_brand_lkup["flag_hashtag"].values)*post_brand_lkup["flag_hashtag"].values)==0
    flag_hashtag = np.array(post_brand_lkup["flag_hashtag"].values, dtype=int)

    # majority voting - useful for initialization
    post_brand_lkup["majority"] = post_brand_lkup["r"].apply(lambda x: np.round(x,0))
    z_init = post_brand_lkup["majority"].values
    z_init = np.array(z_init, dtype=np.int64)
    
    mcmc_in = mcmc_input_class(ii, jj, kk, t_id, ii_lkup, kk_lkup, r_obs, z_init, z_obs, flag_hashtag)
    return mcmc_in

def model_static(mcmc_in, alpha_prior=(1,1), beta_prior=(0,1), asymmetric_accuracy=True, hashtag_treatment="strong", draws=500, tune=500):
    '''
    alpha prior = (mu,sd)
    beta prior = (mu,sd)
    if z_obs is present then oracle
    '''
    model = pm.Model()

    with model:
        if hashtag_treatment=="strong":
            rho_prior = np.ones((2,2))
            rho_prior[1,1] = 49
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup, mcmc_in.flag_hashtag], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)
        if hashtag_treatment=="weak":
            rho_prior = np.ones((2,2))
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup, mcmc_in.flag_hashtag], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)
        elif hashtag_treatment=="oracle" or hashtag_treatment=="none":
            rho_prior = np.ones((1,2))
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)

        beta_prime = pm.Normal('beta_prime', mu=beta_prior[0], sd=beta_prior[1], shape=mcmc_in.K)
        
        if asymmetric_accuracy==True:
            alpha = pm.Normal('alpha', mu=alpha_prior[0], sd=alpha_prior[1], shape=(mcmc_in.J,2))
            def logp(r, z=z, alpha=alpha, beta_prime=beta_prime):
                out = T.switch(T.eq(z[mcmc_in.ii],r),
                               -1*T.log(1+T.exp(-1*alpha[mcmc_in.jj,z[mcmc_in.ii]]*T.exp(beta_prime[mcmc_in.kk]))),
                               -1*alpha[mcmc_in.jj,z[mcmc_in.ii]]*T.exp(beta_prime[mcmc_in.kk]) - 1*T.log(1+T.exp(-1*alpha[mcmc_in.jj,z[mcmc_in.ii]]*T.exp(beta_prime[mcmc_in.kk])))
                               )
                return T.sum(out)            
        else:
            alpha = pm.Normal('alpha', mu=alpha_prior[0], sd=alpha_prior[1], shape=mcmc_in.J)
            def logp(r, z=z, alpha=alpha, beta_prime=beta_prime):
                out = T.switch(T.eq(z[mcmc_in.ii],r),
                               -1*T.log(1+T.exp(-1*alpha[mcmc_in.jj]*T.exp(beta_prime[mcmc_in.kk]))),
                               -1*alpha[mcmc_in.jj]*T.exp(beta_prime[mcmc_in.kk]) - 1*T.log(1+T.exp(-1*alpha[mcmc_in.jj]*T.exp(beta_prime[mcmc_in.kk])))
                               )
                return T.sum(out)
        r = pm.DensityDist('r', logp, observed=mcmc_in.r_obs, shape=len(mcmc_in.r_obs))

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=1)
        
    return trace

def model_dawidskene(mcmc_in, alpha_prior, asymmetric_accuracy=True, hashtag_treatment="strong", draws=500, tune=500):
    '''
    alpha prior = (K,J,2) matrix of pseudocounts for dirichlet
    if z_obs is present then oracle
    '''
    model = pm.Model()

    with model:
        if hashtag_treatment=="strong":
            rho_prior = np.ones((2,2))
            rho_prior[1,1] = 49
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup, mcmc_in.flag_hashtag], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)
        elif hashtag_treatment=="weak":
            rho_prior = np.ones((2,2))
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup, mcmc_in.flag_hashtag], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)
        elif hashtag_treatment=="oracle" or hashtag_treatment=="none":
            rho_prior = np.ones((1,2))
            rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2))
            z = pm.Categorical('z', 
                                p=rho[mcmc_in.kk_lkup], 
                                observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                                testval=mcmc_in.z_init,
                                shape=mcmc_in.N)
        
        if asymmetric_accuracy==True:
            alpha = pm.Dirichlet("alpha", a=alpha_prior, shape=(2,mcmc_in.K,mcmc_in.J,2))
            def logp(r, z=z, alpha=alpha):
                out = T.switch(T.eq(z[mcmc_in.ii],r),
                               T.log(alpha[z[mcmc_in.ii],mcmc_in.kk,mcmc_in.jj,1]),
                               T.log(1-alpha[z[mcmc_in.ii],mcmc_in.kk,mcmc_in.jj,1])
                               )
                return T.sum(out)
        else:
            alpha = pm.Dirichlet("alpha", a=alpha_prior, shape=(mcmc_in.K,mcmc_in.J,2))
            def logp(r, z=z, alpha=alpha):
                out = T.switch(T.eq(z[mcmc_in.ii],r),
                               T.log(alpha[mcmc_in.kk,mcmc_in.jj,1]),
                               T.log(1-alpha[mcmc_in.kk,mcmc_in.jj,1])
                               )
                return T.sum(out)
        r = pm.DensityDist('r', logp, observed=mcmc_in.r_obs, shape=len(mcmc_in.r_obs))

    with model:
        step1 = pm.NUTS(vars=[rho, alpha])
        step2 = pm.CategoricalGibbsMetropolis(vars=[z.missing_values])
        trace = pm.sample(draws=draws, tune=tune, step=[step1, step2], chains=1)
        
    return trace

def model_evolving(mcmc_in, brand_level=True, asymmetric_accuracy=True, draws=500, tune=500):
    interval = 20
    tt_id = mcmc_in.t_id//interval
    # a = np.array([[2,1],[2,1]])
    j_len = (pd.Series(mcmc_in.jj).value_counts()//interval)+1
    j_len_max = max(j_len)

    model = pm.Model()

    with model:
        # the true labels
        rho_prior = np.ones((1,2))
        rho = pm.Dirichlet('rho', a=rho_prior, shape=(mcmc_in.K,2))
        z = pm.Categorical('z', 
                            p=rho[mcmc_in.kk_lkup], 
                            observed=np.ma.masked_values(mcmc_in.z_obs, value=-999),
                            testval=mcmc_in.z_init,
                            shape=mcmc_in.N)

        # credibilities
        if asymmetric_accuracy==False:
            volatility = pm.HalfNormal('volatility', sd=0.75, shape=mcmc_in.J, testval=0.75*T.ones(mcmc_in.J))
            alpha_walk = [T.concatenate([
                    pm.GaussianRandomWalk('alpha_walk{0}'.format(j),  
                    sd=volatility[j],
                    shape=j_len[j], 
                    init=pm.Normal.dist(0.5,1),
                    testval=T.ones(j_len[j])),
                        T.ones(j_len_max-j_len[j])]) for j in range(mcmc_in.J)]
            alpha_walk = T.as_tensor_variable(alpha_walk) 
            if brand_level==True:
                beta_prime = pm.Normal('beta_prime', mu=0, sd=0.4, shape=mcmc_in.K)
                def logp(r, z=z, alpha_walk=alpha_walk):
                    out = T.switch(T.eq(z[mcmc_in.ii],r),
                                  -1*T.log(1+T.exp(-1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk]))),
                                  -1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk]) - 1*T.log(1+T.exp(-1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk])))
                                  )
                    return T.sum(out)
            elif brand_level==False:
                beta_prime = pm.Normal('beta_prime', mu=0, sd=0.4, shape=mcmc_in.N)
                def logp(r, z=z, alpha_walk=alpha_walk):
                    out = T.switch(T.eq(z[mcmc_in.ii],r),
                                  -1*T.log(1+T.exp(-1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii]))),
                                  -1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii]) - 1*T.log(1+T.exp(-1*alpha_walk[mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii])))
                                  )
                    return T.sum(out)

        elif asymmetric_accuracy==True:
            volatility0 = pm.HalfNormal('volatility0', sd=0.75, shape=mcmc_in.J, testval=0.75*T.ones(mcmc_in.J))
            alpha_walk0 = [T.concatenate([
                    pm.GaussianRandomWalk('alpha_walk0{0}'.format(j),  
                    sd=volatility0[j],
                    shape=j_len[j], 
                    init=pm.Normal.dist(0.5,1),
                    testval=T.ones(j_len[j])),
                        T.ones(j_len_max-j_len[j])]) for j in range(mcmc_in.J)]
            alpha_walk0 = T.as_tensor_variable(alpha_walk0) 
            
            volatility1 = pm.HalfNormal('volatility1', sd=0.75, shape=mcmc_in.J, testval=0.75*T.ones(mcmc_in.J))
            alpha_walk1 = [T.concatenate([
                    pm.GaussianRandomWalk('alpha_walk1{0}'.format(j),  
                    sd=volatility1[j],
                    shape=j_len[j], 
                    init=pm.Normal.dist(0.5,1),
                    testval=T.ones(j_len[j])),
                        T.ones(j_len_max-j_len[j])]) for j in range(mcmc_in.J)]
            alpha_walk1 = T.as_tensor_variable(alpha_walk1) 
            
            alpha_walk = T.as_tensor_variable([alpha_walk0,alpha_walk1])
            
            if brand_level==True:
                beta_prime = pm.Normal('beta_prime', mu=0, sd=0.4, shape=mcmc_in.K)
                def logp(r, z=z, alpha_walk=alpha_walk):
                    out = T.switch(T.eq(z[mcmc_in.ii],r),
                                  -1*T.log(1+T.exp(-1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk]))),
                                  -1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk]) - 1*T.log(1+T.exp(-1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.kk])))
                                  )
                    return T.sum(out)
            elif brand_level==False:
                beta_prime = pm.Normal('beta_prime', mu=0, sd=0.4, shape=mcmc_in.N)
                def logp(r, z=z, alpha_walk=alpha_walk):
                    out = T.switch(T.eq(z[mcmc_in.ii],r),
                                  -1*T.log(1+T.exp(-1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii]))),
                                  -1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii]) - 1*T.log(1+T.exp(-1*alpha_walk[z[mcmc_in.ii],mcmc_in.jj,tt_id]*T.exp(beta_prime[mcmc_in.ii])))
                                  )
                    return T.sum(out)
        r = pm.DensityDist('r', logp, observed=mcmc_in.r_obs)    

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=1)  
    
    return trace


def model_selector(model, mcmc_in, brand_level=True, asymmetric_accuracy=True, hashtag_treatment="strong", draws=500, tune=500):
    if model=='dawidskene':
        alpha_prior = np.ones((mcmc_in.K,mcmc_in.J,2))
        alpha_prior[:,:,0] = 1.5
        alpha_prior[:,:,1] = 3.5
        trace = model_dawidskene(mcmc_in,
                             alpha_prior=alpha_prior, 
                             asymmetric_accuracy=asymmetric_accuracy,
                             hashtag_treatment=hashtag_treatment,
                             draws=draws, 
                             tune=tune)
    elif model=='static':
        trace = model_static(mcmc_in, 
                                  alpha_prior=(1,1), 
                                  beta_prior=(0,0.5),
                                  asymmetric_accuracy=asymmetric_accuracy,
                                  hashtag_treatment=hashtag_treatment,
                                  draws=draws,
                                  tune=tune)      
    elif model=='evolving':
        trace = model_evolving(mcmc_in, 
            brand_level=brand_level,
            asymmetric_accuracy=asymmetric_accuracy,
            draws=draws, 
            tune=tune)
    return trace

def store_trace(pkl_name, trace, mcmc_in, ii_transformer, jj_transformer, kk_transformer):
    vars_of_interest = [var for var in trace.varnames if "stickbreaking" not in var]
    trace_dict = {}
    for var in vars_of_interest:
        trace_dict[var] = trace[var]

    with open(pkl_name, 'wb') as f:
        pickle.dump({
                "ii_transformer":ii_transformer, 
                "jj_transformer":jj_transformer, 
                "kk_transformer":kk_transformer,
                "trace":trace_dict,
                "z_obs":mcmc_in.z_obs,
                "z_init":mcmc_in.z_init,
                "flag_hashtag":mcmc_in.flag_hashtag}, f)
    return None

if __name__=="__main__":
    cfg_dict = {}
    cfg_dict["cfg1"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg1.pkl")}
    cfg_dict["cfg2"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg2.pkl")}
    cfg_dict["cfg3"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg3.pkl")}
    cfg_dict["cfg4"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg4.pkl")}
    cfg_dict["cfg5"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":False, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg5.pkl")}
    cfg_dict["cfg6"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":False, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg6.pkl")}
    cfg_dict["cfg7"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":False, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg7.pkl")}
    cfg_dict["cfg8"] = {"MODEL":"evolving", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":False, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg8.pkl")}
    cfg_dict["cfg9"] = {"MODEL":"static", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg9.pkl")}
    cfg_dict["cfg10"] = {"MODEL":"static", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":10, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg10.pkl")}
    cfg_dict["cfg11"] = {"MODEL":"static", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":True, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg11.pkl")}
    cfg_dict["cfg12"] = {"MODEL":"static", "CONDENSE_REPEAT_VOTES":True, "BRAND_LEVEL":True, "ASYMMETRIC_ACCURACY":False, "HASHTAG_TREATMENT":"oracle", "KEEP_PROLIFIC_CUT":20, "MODEL_DECISION_AS_WORKER":False, "DRAWS":500, "TUNE":1000, "TRACE_NAME":os.path.join("output","trace_cfg12.pkl")}

    if len(sys.argv)==1:
        print("please supply a cfg. ---tribe.py cfg1---")
    else:
        cfg = cfg_dict[sys.argv[1]]
        tribe_csvs = [
            os.path.join("input","rohan_11977_1513051779.csv",)
            os.path.join("input","rohan_13584_1513051779.csv",)
            os.path.join("input","rohan_14937_1513051907.csv",)
            os.path.join("input","rohan_15004_1513051907.csv",)
            os.path.join("input","rohan_15336_1513051907.csv",)
            os.path.join("input","rohan_17231_1513051779.csv",)
            os.path.join("input","rohan_17334_1513051779.csv",)
            os.path.join("input","rohan_18792_1513051907.csv",)
            os.path.join("input","rohan_18795_1513051907.csv",)
            os.path.join("input","rohan_19123_1513051779.csv",)
            os.path.join("input","rohan_19141_1513051779.csv",)
            os.path.join("input","rohan_19154_1513051779.csv",)
            os.path.join("input","rohan_19321_1513051779.csv",)
            os.path.join("input","rohan_19491_1513050263.csv",)
            os.path.join("input","rohan_19491_1513051779.csv",)
        ]
        brand_lookup = brand_lkup.gen_brand_lookup()
        tribe_df, validation_df, ii_transformer, jj_transformer, kk_transformer = prep_tribe(tribe_csvs, 
                                                                              brand_lookup, 
                                                                              keep_prolific_cut=cfg["KEEP_PROLIFIC_CUT"], 
                                                                              hashtag_treatment=cfg["HASHTAG_TREATMENT"],
                                                                              condense_repeat_votes=cfg["CONDENSE_REPEAT_VOTES"],
                                                                              model_decision_as_worker=cfg["MODEL_DECISION_AS_WORKER"])
        mcmc_in = get_inputs(tribe_df)

        trace = model_selector(model=cfg["MODEL"], 
                               mcmc_in=mcmc_in, 
                               brand_level=cfg["BRAND_LEVEL"],
                               asymmetric_accuracy=cfg["ASYMMETRIC_ACCURACY"],
                               hashtag_treatment=cfg["HASHTAG_TREATMENT"], 
                               draws=cfg["DRAWS"], 
                               tune=cfg["TUNE"])
        store_trace(cfg["TRACE_NAME"], trace, mcmc_in, ii_transformer, jj_transformer, kk_transformer)
        validation_df.to_csv(os.path.join("output","validation_df.csv"))
