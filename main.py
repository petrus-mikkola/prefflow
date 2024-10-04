import os
import torch
import numpy as np
import normflows as nf
import hydra
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from flows import RealNVP, ResidualFlow, NeuralSplineFlow
from prefflow import PrefFlow
from plotter import Plotter
from target import set_up_problem
from misc import convert_to_ranking, convert_to_ranking_and_change_k, sample_combinations_and_rank, empirical_winner_distribution
from metrics import mean_loglik, wasserstein_dist, statistics, mmtv


# # e.g. by command: python main.py --config-name=onemoon --multirun exp.seed=1,2,3,4,5

@hydra.main(version_base=None, config_path="conf/experiment/mainfigure") #pass experiment (conf) as --config-name argument in terminal
def main(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
 
    if not cfg.plot.showduringtraining: # Show plots?
        matplotlib.use('Agg')

    ### Device and Precision ###
    torch.set_default_dtype(torch.float64 if cfg.device.precision_double else torch.float32)
    #enable_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    #enable_mps = False #Do not enable as MPS framework when precision is float64
    #device = torch.device('mps' if torch.backends.mps.is_available() and enable_mps else 'cpu')
    device = torch.device(cfg.device.device)

    ### Random seeds ###
    import random
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    random.seed(cfg.exp.seed)
    #Below lines prevent pickling
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(cfg.exp.seed)
    #    torch.cuda.manual_seed_all(cfg.exp.seed)  # For multi-GPU.
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False

    ### Target distribution ###
    # Set target prior distirbution expert has in her mind
    # Assume unbounded domain but assume bounding box for sampling
    target_name = cfg.exp.target
    D = cfg.exp.d
    target, bounds, uniform, D, normalize = set_up_problem(target_name,D)

    ### Base distribution ###
    if cfg.params.flow == "gaussianmodel":
        q0 = nf.distributions.AffineGaussian(shape=D,affine_shape=D)
    else:
        q0 = nf.distributions.DiagGaussian(D, trainable=False)
        if target_name in ["llm_prior","abalone_density","abalone_age"]:
            q0.log_scale.fill_(np.log(0.4))

    ### Flow architecture ###
    nflows = cfg.params.nflows
    if cfg.params.flow == "realnvp":
        nfm = RealNVP(nflows,D,q0,device,cfg.device.precision_double)
    if cfg.params.flow == "residualflow":
        nfm = ResidualFlow(nflows,D,q0,device,cfg.device.precision_double)
    if cfg.params.flow == "neuralsplineflow":
        nfm = NeuralSplineFlow(nflows,D,q0,device,cfg.device.precision_double)
    if cfg.params.flow == "gaussianmodel":
        nfm = RealNVP(nflows,D,q0,device,cfg.device.precision_double) #with given network initialization this acts as identity mapping

    ### Data generation part 1 ###
    if target_name not in ["llm_prior","abalone_density","abalone_age"]:
        if target_name == "twomoons": #bimodal target needs two means
            target_mean1 = torch.tensor([-2.0,0.0])
            target_mean2 = torch.tensor([2.0,0.0])
            onemoon,_,_,_,_ = set_up_problem("onemoon",2)
            target_std = onemoon.sample(10000).std(dim=0).double()
        else:
            target_sample = target.sample(10000)
            target_mean = target_sample.mean(dim=0).double()
            target_std = target_sample.std(dim=0).double()
    def sample_alternatives(n,k=2,distribution="uniform"):
        if target_name == "abalone_density":
                num_rows, _ = data.shape
                X = torch.empty(k,D,0)
                for _ in range(n):
                    indices = np.random.choice(num_rows, size=k, replace=False) # Randomly choose k unique indices from the range 0 to num_rows-1
                    selected_rows = data[indices,:]
                    tensor = torch.tensor(selected_rows).unsqueeze(2)
                    X = torch.cat((X,tensor),dim=2)
                return X.squeeze(2)
        else:
            if distribution=="uniform":
                return uniform.sample(torch.tensor([k*n])).to(device)
            elif distribution=="target":
                return target.sample(k*n).to(device)
            elif distribution=="mixture_uniform_gaussian":
                if target_name == "twomoons":
                    means = torch.stack([target_mean1, target_mean2])
                    covariance_matrix = target_std * torch.eye(D).unsqueeze(0).repeat(2, 1, 1)
                    component_distribution = torch.distributions.MultivariateNormal(means, covariance_matrix)
                    mixing_probs = torch.tensor([0.5, 0.5])
                    mixing_distribution = torch.distributions.Categorical(mixing_probs)
                    target_gaussian = torch.distributions.MixtureSameFamily(mixing_distribution, component_distribution)
                else:
                    target_gaussian = torch.distributions.MultivariateNormal(target_mean, target_std*torch.eye(D))
                howoftentarget = cfg.exp.mixture_success_prob
                samples = []  
                for _ in range(k):
                    if np.random.sample() <= howoftentarget:
                        x = target_gaussian.sample((n,))
                    else:
                        x = uniform.sample(torch.tensor([n])).to(device)
                    samples.append(x)
                return torch.cat(samples, dim=0)
    def expert_feedback_pairwise(comp,s=None):
        noise = (0,0) if (s is None) else torch.distributions.Exponential(s).sample((2,)).to(device)
        logprobs = target.log_prob(comp).to(device)
        return torch.ge(logprobs[0] + noise[0],logprobs[1] + noise[1]).long().view(1).to(device)
    def expert_feedback_ranking(alternatives,s=None):
        k = alternatives.shape[0]
        noise = torch.distributions.Exponential(s).sample((k,)).to(device)
        logprobs = target.log_prob(alternatives).to(device) + noise
        _, ranking_inds = torch.sort(logprobs, descending=True)
        return ranking_inds.view(k).to(device)
    def generate_dataset(N,s=None,distribution="uniform"):  #TODO: optimize this, takes long for high D
        X = sample_alternatives(1,2,distribution)
        Y = expert_feedback_pairwise(X,s)
        X = X.unsqueeze(2) #add new dimension, which indicates sample index
        if N > 1:
            for i in range(0,N-1):
                comp = sample_alternatives(1,2,distribution)
                X = torch.cat((X,comp.unsqueeze(2)),2)
                Y = torch.cat((Y,expert_feedback_pairwise(comp,s)),0)
        return X,Y #X.shape = (2,D,N) = (comp,space dimensions, number of comps)
    def generate_dataset_ranking(N,k,s=None,distribution="uniform"):  #TODO: optimize this, takes long for high D
        X = sample_alternatives(1,k,distribution)
        Y = expert_feedback_ranking(X,s).view(1,k)
        X = X.unsqueeze(2) #add new dimension, which indicates sample index
        if N > 1:
            for i in range(0,N-1):
                alternatives = sample_alternatives(1,k,distribution)
                X = torch.cat((X,alternatives.unsqueeze(2)),2)
                Y = torch.cat((Y,expert_feedback_ranking(alternatives,s).view(1,k)),0)
        Xdata = convert_to_ranking(X.numpy(),Y.numpy())
        #return X,Y #X.shape = (k,D,N) = (alternatives,space dimensions, number of rankings)
        return torch.from_numpy(Xdata).view(k,-1,N) 

    ### Data generation part 2 ###
    n = cfg.data.n
    true_s = cfg.exp.true_s
    if target_name in ["llm_prior"]:
        Xdata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataX.npy") #207 rankings
        Ydata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataY.npy", allow_pickle=True)
        Xdata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataX.npy") #13 rankings
        Ydata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataY.npy", allow_pickle=True)
        if cfg.data.k==5: #Basic scenario, matches to k that was used in creating the dataset
            Xdata = convert_to_ranking(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0))
        else:
            Xdata = convert_to_ranking_and_change_k(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0),k=cfg.data.k)
        Xdata = normalize(torch.from_numpy(Xdata))
        variable_names = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
        n = 220
        dataset = Xdata
        ranking = True
    elif target_name in ["abalone_age"]:
        data = np.loadtxt('data/abalone/abalone.data', delimiter=',', usecols=range(1, 9)) #remove first categorical variable
        variable_names = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]
        dataset = data
        n = 4177
        k = 5
        ranking = True
    else:
        if target_name in ["abalone_density"]:
            data = np.loadtxt('data/abalone/abalone.data', delimiter=',', usecols=range(1, 8)) #remove first categorical variable and last output variable
            variable_names = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]
        ranking = True if cfg.data.k > 2 else False #TODO: check this if we want to generalize
        if ranking:
            k = cfg.data.k
            dataset = generate_dataset_ranking(N=n,k=k,s=true_s,distribution=cfg.exp.lambda_dist)
        else:
            dataset = generate_dataset(N=n,s=true_s,distribution=cfg.exp.lambda_dist)
            
    def minibatch(dataset,batch_size,ranking):
        if target_name in ["abalone_age"]:
            Xbatch = sample_combinations_and_rank(dataset,n=batch_size,k=5,d=D)
            batch = normalize(Xbatch)
        else:
            indices = torch.randperm(n)[:batch_size]
            batch = (dataset[0][:,:,indices],dataset[1][indices]) if not ranking else dataset[:,:,indices]
        return batch

    #Initialize preferential flow
    prefflow = PrefFlow(nfm,D=D,s=cfg.modelparams.s,ranking=ranking,device=device,precision_double=cfg.device.precision_double)

    #Initialize optimizer
    loss_hist = np.array([])
    batch_size = cfg.params.batch_size
    optimizer = getattr(torch.optim, cfg.params.optimizer.capitalize())
    
    if cfg.params.flow == "gaussianmodel":
        q0_params = list(q0.parameters())
        optimizer_prefflow = optimizer(q0_params,lr=cfg.params.lr, weight_decay=cfg.params.weight_decay) #lr=1e-3
    else:
        optimizer_prefflow = optimizer([{'params':prefflow.parameters()}],lr=cfg.params.lr, weight_decay=cfg.params.weight_decay)

    #Flowsampling
    if target_name not in ["llm_prior","abalone_age"]:
        if target_name in ["abalone_density"]:
            targetsample, targetsample_logprob = target.sample_stable(cfg.plot.nsamples)
        else:
            targetsample = target.sample(cfg.plot.nsamples)
    elif target_name in ["abalone_age"]:
        targetsample = empirical_winner_distribution(data,cfg.plot.nsamples,0)
    def sample_flow(prefflow):
        prefflow.eval()
        flowsample, flowsample_logprob = prefflow.sample_stable(cfg.plot.nsamples)
        flowsample = flowsample.detach()
        flowsample_logprob = flowsample_logprob.detach()
        prefflow.train()
        if target_name in ["llm_prior","abalone_age"]:
            flowsample = normalize(flowsample,reverse=True)
        return flowsample, flowsample_logprob
    initial_flowsample, initial_flowsample_logprob = sample_flow(prefflow) if target_name not in ["llm_prior"] else (None,None)
    initial_loglik = mean_loglik(n,prefflow,minibatch,dataset,ranking)

    plotter = Plotter(D,bounds)



    ### SGD: FS-MAP ###

    for it in tqdm(range(cfg.params.max_iter),disable=not cfg.plot.progressbar_show):
        
        #Sample minibacth
        batch = minibatch(dataset,batch_size,ranking)

        #Update flow parameters
        prefflow.train()
        optimizer_prefflow.zero_grad()
        loss = -prefflow.logposterior(batch,cfg.modelparams.weightprior)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer_prefflow.step()

        if cfg.params.flow == "residualflow":
            # When using ResidualFlow, make layers Lipschitz continuous
            nf.utils.update_lipschitz(prefflow, 50)

        loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())

        # Plot learned posterior
        if (it + 1) % cfg.plot.show_iter == 0:
            print("loss: " + str(loss.to('cpu').detach().numpy()))
            if target_name in ["onemoon","twomoons"]:
                if cfg.plot.showdatapoints:
                    showdata = minibatch(dataset,batch_size=n,ranking=ranking)
                    probmassinarea = plotter.plot_moon(target,prefflow,data=showdata,cfg=cfg)
                else:
                    probmassinarea = plotter.plot_moon(target,prefflow,data=None,cfg=cfg)
                print("probmass in domain: " + str(probmassinarea))
                plt.show()
            if target_name in ["funnel","onegaussian","twogaussians","llm_prior","banana"]:
                flowsample, _ = sample_flow(prefflow)
                plotter.plot_dist(flowsample)

    
    
    
    
    ############ Reporting and plotting the results #################
    
    #Experiment name
    def experiment_name():
        terms = list(range(10))
        terms[0] = target_name
        terms[1] = cfg.exp.lambda_dist
        terms[2] = str(n)
        terms[3] = "maxiter" + str(cfg.params.max_iter)
        terms[4] = "flows" + str(cfg.params.nflows)
        terms[5] = "bsize" + str(batch_size)
        terms[6] = "true_s" + str(true_s)
        terms[7] = "lr" + str(cfg.params.lr)
        terms[8] = "prior_weight" + str(cfg.modelparams.weightprior)
        terms[9] = "seed" + str(cfg.exp.seed)
        if cfg.exp.exp_id is None:
            expname = str(D) + "D"
        else:
            expname = cfg.exp.exp_id + "_" + str(D) + "D"
        for t in terms:
            expname += "_" + str(t)
        return expname

    #Save optimized hyperparameters
    def save_hyperparameters_log():
        f = open(os.path.join(output_folder,"hyperparameters_"+ experiment_name() + ".txt"), "w")
        f.write("Hyperparameters \n")
        f.write("s: " + str(prefflow.s.to('cpu').detach().numpy())+"\n")
        f.close()

    save_hyperparameters_log()
    
    #Plot loss trajectory
    plt.figure(figsize=(15, 15))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder,"loss_"+ experiment_name() + ".png"), dpi=150)
    plt.show()

    #Flowsampling
    flowsample, flowsample_logprob = sample_flow(prefflow)

    #Report results
    if target_name in ["funnel","onegaussian","twogaussians","llm_prior","abalone_density","abalone_age","banana"]:
        np.save(os.path.join(output_folder,'flowsamples.npy'), flowsample.numpy())
    f = open(os.path.join(output_folder,"results_"+ experiment_name() + ".txt"), "w")
    f.write("Results after the learning has finished.\n")
    f.write("Last iteration loss: " + str(loss.to('cpu').detach().numpy()) +"\n")
    f.write("Initial full data average log-likelihood: " + str(initial_loglik) +"\n")
    final_loglik = mean_loglik(n,prefflow,minibatch,dataset,ranking)
    f.write("Final full data average log-likelihood: " + str(final_loglik) +"\n")
    if target_name not in ["llm_prior"]:
        Wd_init = wasserstein_dist(initial_flowsample[:cfg.plot.wasserstein_nsamples,:],targetsample[:cfg.plot.wasserstein_nsamples,:])
        f.write("Initial Wasserstein distance between the target and the flow: " + str(Wd_init)+ "\n")
        Wd = wasserstein_dist(flowsample[:cfg.plot.wasserstein_nsamples,:],targetsample[:cfg.plot.wasserstein_nsamples,:])
        f.write("Final Wasserstein distance between the target and the flow: " + str(Wd)+ "\n")
        if target_name in ["abalone_age"]:
            tv_init = 0
            tv = 0
        else:
            tv_init = mmtv(initial_flowsample,targetsample)
            tv = mmtv(flowsample,targetsample)
        f.write("Initial mean marginal total variation distance between the target and the flow: " + str(tv_init)+ "\n")
        f.write("Final mean marginal total variation distance between the target and the flow: " + str(tv)+ "\n")
        results = np.array([[initial_loglik,final_loglik],[Wd_init,Wd],[tv_init,tv]])
        np.save(os.path.join(output_folder,'results.npy'), results)
    if target_name in ["llm_prior"]:
        f.write(str(statistics(flowsample,variable_names))+"\n")
    f.close()

    #Plot flow
    if target_name in ["onemoon","twomoons"]:
        #Plot learned posterior distribution
        probmassinarea = plotter.plot_moon(target,prefflow,data=None,cfg=cfg)
        #plt.text(x=0.95, y=0.95, s=str(probmassinarea) + '% of the flow density', fontsize=22, color='white', verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(output_folder,experiment_name() + ".png"), dpi=150) #pdf produces poor looking aliasing
        plt.show()
        #Plot learned posterior distribution with datapoints
        showdata = minibatch(dataset,batch_size=n,ranking=ranking)
        probmassinarea = plotter.plot_moon(target,prefflow,data=showdata,cfg=cfg)
        f = open(os.path.join(output_folder,"results_"+ experiment_name() + ".txt"), "a")
        f.write("Probability mass in the domain based on flow density: " + str(probmassinarea)+"%"+"\n")
        f.close()
        #plt.text(x=0.95, y=0.95, s=str(probmassinarea) + '% of the flow density', fontsize=22, color='white', verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(output_folder,"datapoints_" + experiment_name() + ".png"), dpi=150) #pdf produces poor looking aliasing
        plt.show()
    if target_name in ["funnel","onegaussian","twogaussians","abalone_density","abalone_age","banana"]:
        linewidth = 0.3 if target_name == "funnel" else 0.1
        labels = variable_names if target_name in ["abalone_density"] else None
        plotter.plot_dist(flowsample,targetsample,save=True,path=os.path.join(output_folder,experiment_name() + "_targetdisplayed" + ".png"),linewidth=linewidth,labels=labels)
        plotter.plot_dist(flowsample,targetsample,save=True,path=os.path.join(output_folder,experiment_name() + "_targetdisplayed_nomarginal" + ".png"),linewidth=linewidth,marginal_plot_dist2=False,labels=labels)
        plotter.plot_dist(flowsample,None,save=True,path=os.path.join(output_folder,experiment_name() + ".png"),labels=labels)
        #Plot target distribution
        if target_name not in ["abalone_age"]:
            plotter.plot_dist(targetsample,None,save=True,path=os.path.join(output_folder,"target_" + str(D) + "D" + target_name + ".png"))
    if target_name in ["llm_prior"]:
        plotter.plot_dist(flowsample,None,save=True,path=os.path.join(output_folder,experiment_name() + ".png"),labels=variable_names)

        

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()