import torch
import numpy as np
from matplotlib import pyplot as plt
import corner

class Plotter():
    
    """
    Class for handling plotting of preferential datapoints and plotting any distribution
    """

    def __init__(self, d, bounds):
        self.d = d  # Dimension of the distribution
        self.bounds = bounds
    
    def plot_data(self,batch):
        batchX = batch[0]
        batchY = batch[1]
        for i in range(0, batchX.shape[2]):
            if batchY[i]:
                thetaprime = batchX[0,:,i]
                thetaprimeprime = batchX[1,:,i]
            else:
                thetaprime = batchX[1,:,i]
                thetaprimeprime = batchX[0,:,i]
            plt.plot(thetaprime[0], thetaprime[1], 'r+', markersize=25)  # Pref point = red+   (if want bolded use 'rP')
            plt.plot(thetaprimeprime[0], thetaprimeprime[1], 'b_' , markersize=25)  # Non-Pref point = blue-


    def plot_ranking_data(self,batch):
        #batchX = batch[1]
        k,D,N = batch.shape
        markers = [str(mark) for mark in range(1,k+1)]
        for i in range(N):
            for j in range(k):
                x = batch[j,0,i]
                y = batch[j,1,i]
                color = "red" if j==0 else "blue"
                plt.text(x, y, markers[j], color=color, fontsize=25)
    
    def plot_moon(self,target,prefflow,data,cfg):
        xx, yy = torch.meshgrid(torch.linspace(self.bounds[0][0], self.bounds[0][1], cfg.plot.grid_size), torch.linspace(self.bounds[1][0], self.bounds[1][1], cfg.plot.grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.double().to(cfg.device.device) if cfg.device.precision_double else zz.float().to(cfg.device.device)
        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        prob_target = torch.exp(log_prob)
        if prefflow is None:
            plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
            return None
        prefflow.eval()
        log_prob = prefflow.log_prob(zz).to('cpu').view(*xx.shape)
        prefflow.train()
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        rectangle_vol = ((self.bounds[0][1]-self.bounds[0][0])/cfg.plot.grid_size)*((self.bounds[1][1]-self.bounds[1][0])/cfg.plot.grid_size)
        probmassinarea = round(100*rectangle_vol*torch.sum(prob).detach().numpy(),1)
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.detach().numpy())
        plt.contour(xx, yy, prob_target.detach().numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        if data is not None:
            if cfg.data.k > 2:
                self.plot_ranking_data(data)
            else:
                self.plot_data(data) #plot whole data set
        plt.gca().set_aspect('equal', 'box')
        return probmassinarea 

    def plot_dist(self,dist1_samples,dist2_samples=None,save=False,path=None,nbins=500,nlevels=3,linewidth=0.3,marginal_plot_dist2=True,density_marginal=False,labels=None):
        
        if isinstance(dist1_samples, torch.Tensor):
            dist1_samples = dist1_samples.numpy()

        data = dist1_samples
        if dist2_samples is not None:
            if isinstance(dist2_samples, torch.Tensor):
                dist2_samples = dist2_samples.numpy()
            data2 = dist2_samples


        # Use corner to create the initial corner plot framework
        try:
            figure = corner.corner(data,bins=nbins,density=density_marginal) #bins affect only to marginal plots in the diagonal
        except:
            print("Error in plotting corner plot. Input data shape: " + str(data.shape))

        axes = np.array(figure.axes).reshape((self.d, self.d))
        custom_limits = list(self.bounds)

        for i in range(self.d):
            for j in range(i):
                ax = axes[i, j]
                ax.cla() 
                
                x = data[:, j]
                y = data[:, i]
                
                # Compute the histogram/density
                hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, density=True)
                
                # Generate the meshgrid for plotting
                X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
                
                #Plot heatamp
                ax.pcolormesh(X, Y, hist, shading='auto')

                if dist2_samples is not None:
                    #Plot countour plot
                    x = data2[:, j]
                    y = data2[:, i]

                    hist2, xedges, yedges = np.histogram2d(x, y, bins=nbins, density=True)

                    # Generate the grid centers from the edges
                    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
                    y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
                    X, Y = np.meshgrid(x_bin_centers, y_bin_centers)

                    # Assuming you want to keep n levels but exclude the outermost
                    # First, find the max value in hist2 to define your levels
                    max_val = np.max(hist2)
                    # Create levels that exclude the lowest level, e.g., by using np.linspace to divide the range
                    # Adjust the number of levels as desired
                    levels = np.linspace(0, max_val, nlevels+2)[1:]  # This creates n levels excluding the lowest one

                    # Create the contour plot
                    ax.contour(X, Y, hist2.T, cmap=plt.get_cmap('cool'), linewidths=linewidth, levels=levels)

        for i in range(self.d):
            for j in range(self.d):
                ax = axes[i, j]
                if i == j:  # Diagonal plots (histograms)
                    ax.set_xlim(custom_limits[i])
                    #ax.set_ylim() #TODO: check this
                    if (dist2_samples is not None) and (marginal_plot_dist2):
                        # Plot dist2's marginal distribution on the diagonal
                        x = data2[:, i]  # Data from dist2 for this dimension
                        # Create histogram of dist2 data on the same axis
                        #ax.hist(x, bins=nbins, density=True, histtype='step', color='purple', alpha=0.5)  # `alpha` for transparency
                        n, bins, patches = ax.hist(x, bins=nbins, density=density_marginal, histtype='step', color='magenta', linewidth=1, alpha=0.75) #TODO: check this
                elif j < i:  # Lower triangle
                    ax.set_xlim(custom_limits[j])
                    ax.set_ylim(custom_limits[i])
                # If you decide to customize the upper triangle, you would add an 'elif j > i:' block here

        #manually set labels
        if labels is None:
            labels = [f"x{i}" for i in range(1, self.d + 1)]
        for i in range(self.d):
            for j in range(self.d):
                ax = axes[i, j]
                if j < i:  # Lower triangle
                    # Set y-axis labels for the leftmost column
                    if j == 0:
                        ax.set_ylabel(labels[i])
                    # Set x-axis labels for the bottom row
                    if i == self.d - 1:
                        ax.set_xlabel(labels[j])

                if i == j:  # Diagonal
                    # Set x-axis labels for the diagonal plots
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

        # Adjust the figure and show the plot
        plt.tight_layout()

        if save:
            plt.savefig(path, dpi=150)
                
        plt.show()
                        
            