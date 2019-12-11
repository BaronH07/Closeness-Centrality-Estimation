def calculate_average_euclidean_following(k = 100, iters = 20, g = following_network, sampling_method='random', true_values=following_centralities_selected_nodes): #Calculate the average euclidean distance given k
    """k: the number of pivots(samples)
        iters: number of iterations
        g: the graph
    """
    nodes_g = list(g.nodes())
    nodes_to_estimate = list(true_values.keys())
    n =len(nodes_g) #number of nodes
    true_closeness_values = [v for k,v in sorted(true_values.items())]
    #estimation_closeness_centralities = []
    euclidean_distances = []
    
    for i in range(iters):
        #print("Iteration: ", i+1)
        eccs_i = {} #estimation of ith iteration
        
        if sampling_method == 'random': #Random sampling
            pivots = np.random.choice(nodes_g, k, replace=False)
        elif sampling_method == 'randeg': #Random degree
            degree_distribution = np.array([v for k,v in sorted(dict(g.degree()).items())])
            degree_distribution = degree_distribution/sum(degree_distribution)
            pivots = np.random.choice(nodes_g, k, replace=False, p=degree_distribution)
        elif sampling_method == 'maxmin': #use MaxMin
            pivots = maxmin(k=k, g=g)
        elif sampling_method == 'maxsum':
            pivots =maxsum(k=k, g=g)
        elif sampling_method == 'minsum':
            pivots = minsum(k=k, g=g)
            
        for node in nodes_to_estimate:
            d_pn = 0 #Sum of distances from pivot to node
            n_pa = 0 #number of pivots which have path to the node
            for p in pivots:
                if nx.has_path(g, source=p, target=node) and node!=p:
                    n_pa += 1
                    d_pn += nx.shortest_path_length(g, source=p, target=node)#Sum of distances from pivot to node
            
            if n_pa == 0:
                ecc = 0
            else:
                ecc = (((n_pa/k)*n-1)/(n-1))*(n_pa/d_pn) #Estimated Closeness Centrality of the node
            eccs_i.update({node:ecc})
            
        eccs_i_values = [v for k, v in sorted(eccs_i.items())]
        euclidean_distances.append(euclidean(eccs_i_values, true_closeness_values)/n)
        print("Iteration: ", i+1, " euclidean: ", euclidean(eccs_i_values, true_closeness_values)/n)
        
    return np.array(euclidean_distances).mean()
