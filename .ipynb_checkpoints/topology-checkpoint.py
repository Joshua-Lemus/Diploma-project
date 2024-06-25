import numpy as np
import networkx as nx
import torch
import torch.nn as nn


############ POSITIONS ############

def compute_dist(dim, pos0, pos1, lens=[1,1], torus=True, square=False):
    if torus:
        # If the topology is a torus, distance between opposite edges of a same row is 1
        if dim == 1:
            # Distance is Dij= min(|xi-xj|, N-|xi-xj|)
            dists = torch.min(torch.abs(pos0[:,None] - pos1[None,:]),
                              lens[0] - torch.abs(pos0[:,None] - pos1[None,:]))
            if square:
                dists = dists**2
        elif dim == 2:
            # Do for each dimension
            dists_x = torch.min(torch.abs(pos0[:,0,None] - pos1[None,:,0]),
                                lens[0] - torch.abs(pos0[:,0,None] - pos1[None,:,0]))
            dists_y = torch.min(torch.abs(pos0[:,1,None] - pos1[None,:,1]),
                                lens[1] - torch.abs(pos0[:,1,None] - pos1[None,:,1]))
            dists = dists_x**2 + dists_y**2
            if not square:
                dists = torch.sqrt(dists + 1e-20)
    else: #(not a torus, distance between opposite edges is N)
        if dim == 1:
            if square:
                # Distance is Dij=|xi-xj|
                dists = (pos0[:,None] - pos1[None,:])**2
            else:
                dists = torch.abs(pos0[:,None] - pos1[None,:])
        else:
            if square:
                dists = ((pos0[:,None] - pos1[None,:])**2).sum(2)
            else:
                dists = torch.sqrt(((pos0[:,None] - pos1[None,:])**2).sum(2))
    return dists


def gen_lattice(dim=1, lens=[1,1], torus=True, random_pos=False, tot_points=[10,10], centered=True):
    if random_pos:
        # Create random (uniformly distributed) tensor of dimension [tot_points(=N), dim]
        pos = torch.stack([lens[d] * torch.rand(np.prod(tot_points)) for d in range(dim)], dim=1) #Will just use lens[0]
        if dim == 1:
            # Don't know why this is needed, anyway dim is 1 so the tensor is flat(?)
            pos = pos.flatten()
    else: # (Not random)
        if dim == 1:
            if centered:
                # A tensor going from 0 to lens(=N), excluding 0; step size is (lens/(tot_points+1))(=0.99)
                pos = torch.tensor(np.linspace(0, lens[0], tot_points[0]+1, endpoint=False)[1:], dtype=torch.float)
            else:
                # A tensor going from 0 to lens(=N)-1, step size is (lens/tot_points)(=1)
                pos = torch.arange(0, lens[0], lens[0] / tot_points[0], dtype=torch.float)
        elif dim == 2:
            # Same in 2D, with each dimension stacked in a tensor
            xys = []
            for d in range(2):
                if centered:
                    # A tensor going from 0 to lens(=sqrt(N)), excluding 0; step size is (lens/(tot_points+1))(=sqrt(N/(N+1)))
                    p = torch.tensor(np.linspace(0, lens[d], tot_points[d]+1, endpoint=False)[1:], dtype=torch.float)
                else:
                    # A tensor going from 0 to lens(=sqrt(N))-1; step size is (lens/tot_points)(=sqrt(N/N)=1)
                    p = torch.arange(0, lens[d], lens[d] / tot_points[d], dtype=torch.float)
                xys.append(p)
            x_grid, y_grid = torch.meshgrid(xys)
            pos = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
    return pos


def gen_lattice_banks(dim=1, len_lattice=28, delta_lattice=5, random_pos=False, tot_points=10):
    if random_pos:
        sites = len_lattice * torch.rand(tot_points)
    else:
        sites = torch.arange(0, len_lattice**dim, delta_lattice).type(torch.float)
    if dim == 2:
        pos = torch.stack((sites // len_lattice, sites % len_lattice), dim=1)
    else:
        pos = sites
    return pos


# # DEPRECATED
# def gen_positions(N, dim, lens, tot_points, scale_len,
#                   torus = False,
#                   random_pos = False,
#                   square_M_reg_L1 = False,
#                   shuffle_M_reg = False,
#                   average_M_reg = False,
#                   train_pos=False,
#                   device=torch.device("cpu")):
    
#     pos0 = gen_lattice(dim=dim, lens=lens, torus=torus, random_pos=False, tot_points=[N,N])
#     pos = gen_lattice(dim=dim, lens=scale_len * np.array(lens), torus=torus, random_pos=random_pos, tot_points=tot_points)
#     M_reg_inp, M_reg_rec = None, None
#     if not train_pos:
#         M_reg_inp = compute_dist(dim, pos0, pos, lens=lens, torus=torus).to(device)
#         M_reg_rec = compute_dist(dim, pos, pos, lens=lens, torus=torus).to(device)
#         if square_M_reg_L1:
#             M_reg_inp = M_reg_inp**2
#             M_reg_rec = M_reg_rec**2
#         if shuffle_M_reg:
#             for k in range(shuffle_M_inp.shape[1]):
#                 M_reg_inp[:,k] = M_reg_inp[np.random.permutation(M_reg_inp.shape[0]), k]
#                 M_reg_rec[:,k] = M_reg_rec[np.random.permutation(M_reg_rec.shape[0]), k]
#         if average_M_reg:
#             for k in range(M_reg_inp.shape[1]):
#                 M_reg_inp[:,k] = M_reg_inp[:,k].mean()
#                 M_reg_rec[:,k] = M_reg_rec[:,k].mean()
            
#     return pos0, pos, M_reg_inp, M_reg_rec
    

    
# params_nx = {}
# params_nx['graph_type'] = graph_type
# params_nx['K'] = K
# params_nx['r'] = r
# params_nx['h'] = h
# params_nx['Nbranch'] = Nbranch
# params_nx['degree'] = degree
# params_nx['rewiring_p'] = rewiring_p
# params_nx['m'] = m
# params_nx['num_long_range'] = num_long_range
# params_nx['prob_decay_exp'] = prob_decay_exp
# params_nx['dim'] = dim
# params_nx['seed'] = seed 

# G, adj = nt.get_network_structure(params_nx)

# # NETWORK STRUCTURE

# draw_graph = True

# # graph_type = 'balanced_tree'
# # graph_type = 'random_regular'
# graph_type = 'watts_strogatz'
# # graph_type = 'barabasi_albert'
# # graph_type = 'navigable_small_world'
# # graph_type = 'soft_random_geometric'

# if graph_type == 'balanced_tree':
#     Nbranch = 3
#     h = 3
#     r = 3
#     G = nx.balanced_tree(r,h)
#     for i in range(Nbranch-1):
#         gi = nx.balanced_tree(r,h)
#         rooti = np.random.randint(0,len(G.nodes))
#         rootj = np.random.randint(0,len(gi.nodes))
#         G = nx.join([(G,rooti),(gi,rootj)])
# elif graph_type == 'random_regular':
#     degree = 5
#     G = nx.generators.random_graphs.random_regular_graph(degree, K)
# elif graph_type == 'watts_strogatz':
#     degree = 4
#     rewiring_p = 0.2
#     G = nx.generators.random_graphs.watts_strogatz_graph(K, degree, rewiring_p)
# elif graph_type == 'barabasi_albert':
#     m = 3
#     G = nx.generators.random_graphs.barabasi_albert_graph(K, m)
# elif graph_type == 'navigable_small_world':
#     connection_diameter = 1
#     num_long_range = 1
#     prob_decay_exp = 0
#     G = nx.generators.geometric.navigable_small_world_graph(sqK,
#                                                             p=connection_diameter,
#                                                             q=num_long_range,
#                                                             r=prob_decay_exp,
#                                                             dim=dim,
#                                                             seed=seed)
# # TODO
# # elif graph_type == 'soft_random_geometric':
# #     radius = 10
# #     scale = 10
# #     p_dist = lambda dist : np.exp(-dist/scale)
# #     pos_dict = {i: tuple(pos) for i, pos in enumerate(pos.cpu().numpy())}
# #     G = nx.generators.geometric.soft_random_geometric_graph(K, radius,
# #                                                             dim=dim,
# #                                                             pos=pos_dict,
# #                                                             p=2,
# #                                                             p_dist=p_dist,
# #                                                             seed=seed)
    
# nodes, edges = list(G.nodes), list(G.edges)
# print(f"(nodes, edges)  = {len(nodes), len(edges)}")

# if nx.is_tree(G):
#     print("It's a tree!")
# if not nx.is_directed(G):
#     if not nx.is_connected(G):
#         print('unconnected graph!')
    
# # adj =  nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
# adj = nx.to_numpy_matrix(G)



# if draw_graph:
#     pos = graphviz_layout(G, prog='neato')
#     nx.draw(G, pos, node_size=30, alpha=0.8, node_color='#5994f5', with_labels=False)
#     plt.show()
    
    
def get_network_structure(params):
    graph_type = params['graph_type']
    seed = params['seed']
    
    if graph_type == 'balanced_tree':
        r = params['r']
        h = params['h']
        Nbranch = params['Nbranch']
        G = nx.balanced_tree(r, h)
        for i in range(Nbranch - 1):
            gi = nx.balanced_tree(r,h)
            rooti = np.random.randint(0,len(G.nodes))
            rootj = np.random.randint(0,len(gi.nodes))
            G = nx.join([(G,rooti),(gi,rootj)])
    elif graph_type == 'random_regular':
        degree = params['degree']
        N = params['N']
        G = nx.generators.random_graphs.random_regular_graph(degree, N, seed=seed)
    elif graph_type == 'watts_strogatz':
        N = params['N']
        degree = params['degree']
        rewiring_p = params['rewiring_p']
        G = nx.generators.random_graphs.watts_strogatz_graph(N, degree, rewiring_p, seed=seed)
    elif graph_type == 'barabasi_albert':
        N = params['N']
        m = params['m']
        G = nx.generators.random_graphs.barabasi_albert_graph(N, m, seed=seed)
    elif graph_type == 'navigable_small_world':
        sqN = params['sqN']
        connection_diameter = params['connection_diameter']
        num_long_range = params['num_long_range']
        prob_decay_exp = params['prob_decay_exp']
        dim = params['dim']
        seed = params['seed']
        G = nx.generators.geometric.navigable_small_world_graph(sqN,
                                                                p=connection_diameter,
                                                                q=num_long_range,
                                                                r=prob_decay_exp,
                                                                dim=dim,
                                                                seed=seed)
    # TODO
#     elif graph_type == 'soft_random_geometric':
#         radius = 10
#         scale = 10
#         p_dist = lambda dist : np.exp(-dist/scale)
#         pos_dict = {i: tuple(pos) for i, pos in enumerate(pos.cpu().numpy())}
#         G = nx.generators.geometric.soft_random_geometric_graph(K, radius,
#                                                                 dim=dim,
#                                                                 pos=pos_dict,
#                                                                 p=2,
#                                                                 p_dist=p_dist,
#                                                                 seed=seed)
    
    nodes, edges = list(G.nodes), list(G.edges)
    print(f"(nodes, edges)  = {len(nodes), len(edges)}")

    if nx.is_tree(G):
        print("It's a tree!")
    if not nx.is_directed(G):
        if not nx.is_connected(G):
            print('unconnected graph!')
    
    # adj =  nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
    adj = nx.to_numpy_matrix(G)
    
    return G, adj
