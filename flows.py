import torch
import normflows as nf

def RealNVP(K,D,q0,device,PRECISION_DOUBLE):
    latent_size = D
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]
    # Construct flow model
    if PRECISION_DOUBLE:
        nfm = nf.NormalizingFlow(q0=q0,flows=flows)
    else:
        nfm = nf.NormalizingFlow(q0=q0.float(), flows=[flow.float() for flow in flows]) #force float32 as float64 does not work with MPS
    # Move model on MPS/CUDA if available
    nfm = nfm.to(device)
    nfm = nfm.double() if PRECISION_DOUBLE else nfm.float()
    # Initialize ActNorm
    nfm.eval()
    z, _ = nfm.sample(num_samples=2 ** (D+5)) #z, _ = nfm.sample(num_samples=2 ** 7) default number of samples for act norm
    nfm.train()
    z_np = z.to('cpu').detach().numpy()
    #plt.figure(figsize=(15, 15))
    #plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[0][1]]])
    #plt.gca().set_aspect('equal', 'box')
    #plt.show()
    return nfm

def ResidualFlow(K,D,q0,device,PRECISION_DOUBLE):
    latent_size = D
    hidden_units = 50 #must use fewer number of hiddenunits than default 128 as memory runs out
    hidden_layers = 3
    flows = []
    for i in range(K):
       net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                  init_zeros=True, lipschitz_const=0.9)
       flows += [nf.flows.Residual(net, reduce_memory=True)]
       flows += [nf.flows.ActNorm(latent_size)]
    # Construct flow model
    if PRECISION_DOUBLE:
        nfm = nf.NormalizingFlow(q0=q0,flows=flows)
    else:
        nfm = nf.NormalizingFlow(q0=q0.float(), flows=[flow.float() for flow in flows]) #force float32 as float64 does not work with MPS
    # Move model on MPS/CUDA if available
    nfm = nfm.to(device)
    nfm = nfm.double() if PRECISION_DOUBLE else nfm.float()
    # Initialize ActNorm
    nfm.eval()
    z, _ = nfm.sample(num_samples=2 ** (D+5)) #z, _ = nfm.sample(num_samples=2 ** 7) default number of samples for act norm
    nfm.train()
    z_np = z.to('cpu').detach().numpy()
    #plt.figure(figsize=(15, 15))
    #plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[0][1]]])
    #plt.gca().set_aspect('equal', 'box')
    #plt.show()
    return nfm


def NeuralSplineFlow(K,D,q0,device,PRECISION_DOUBLE):
    latent_size = D
    hidden_units = 128
    hidden_layers = 2 
    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    # Construct flow model
    if PRECISION_DOUBLE:
        nfm = nf.NormalizingFlow(q0=q0,flows=flows)
    else:
        nfm = nf.NormalizingFlow(q0=q0.float(), flows=[flow.float() for flow in flows]) #force float32 as float64 does not work with MPS
    # Move model on MPS/CUDA if available
    nfm = nfm.to(device)
    nfm = nfm.double() if PRECISION_DOUBLE else nfm.float()
    return nfm