'''
python file about loss used in model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mtmcat_loss(inputs, clinical, loss_log_out=True, loss_mode='ce', params=dict()):
    '''
    input: [tensor(a1_proba, a2_proba, ...)]
    clinical: [tensor(label)]
    input and clinical in torch.device
    return loss in torch.device
    '''
    if loss_mode == 'ce':
        loss_tool = nn.CrossEntropyLoss()
    elif loss_mode == 'focal':
        loss_tool = FocalLoss(params=params)
    target = clinical[0]
    output = inputs[0].unsqueeze(0)  # shape: (1, class)
    loss = loss_tool(output, target.long())
    return loss

# cross entropy sur loss
def cross_entropy_sur_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    # hazards: tensor shape: (1, n_class)
    # Y: tensor shape: (1,1)
    # S: tensor shape:(1, n_class)
    # c: tensor shape:(1)

    batch_size = len(Y)    # will get batch_size=1
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

#nll sur loss
def nll_sur_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 1, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, params=dict()):
        super(FocalLoss, self).__init__()
        alpha = float(params['focal_alpha'])
        gamma = float(params['focal_gamma'])
        self.device = params['device']
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(self.device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_cpu = targets.clone().cpu()
        targets_one_hot = torch.zeros(inputs.shape[0], 2).scatter_(1, targets_cpu.unsqueeze(0), 1).to(self.device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        # two stage classification
        targets_one_hot = targets_one_hot.long()
        at = self.alpha.gather(0, targets_one_hot.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        # print(F_loss)
        # print(F_loss.sum())
        return F_loss.sum()


# # cox log partial loss
def neg_log_partial_likelihood(e_true, pred):
    # from https://github.com/luckydoggy98/SAEsurv-net/blob/main/SAEsurv_net/SAEsurv_net.py
    # need sort survival time with index!!!
    # assume that all samples have the same baseline hazard function
    # need more than 1 sample in an mini-batch?
    n_obs = e_true.sum()

    hazard = torch.exp(pred)
    cost = torch.negative(
        torch.divide(
            (
                torch.multiply(
                    e_true, torch.subtract(
                        pred, torch.log( 
                            torch.cumsum(torch.flip(hazard, dims = [-1]), dim = -1))))).sum(),
                            n_obs))
    
    return cost

def neg_log_partial_likelihood_with_reset_index(e_true, pred):
    # need more than 1 sample in an mini-batch
    e_true = e_true.view(-1)
    pred = pred.view(-1)
    index = torch.argsort(e_true, dim = -1)
    return neg_log_partial_likelihood(e_true[index], pred[index])



def encode_survival(time,
                    event,
                    bins):
    """
    from https://github.com/mkazmier/torchmtlr/blob/master/torchmtlr/utils.py
    Encodes survival time and event indicator in the format
    required for MTLR training.
    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.
    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.
    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def MTLR_survival_loss(y_pred, y_true, E, tri_matrix, reduction='mean'):
    """
    Compute the MTLR survival loss
    from https://github.com/AstraZeneca/multitask_impute/blob/master/OmiEmbed/models/losses.py
    https://cran.r-project.org/web/packages/MTLR/vignettes/workflow.html
    """
    # Get censored index and uncensored index
    censor_idx = []
    uncensor_idx = []
    for i in range(len(E)):
        # If this is a uncensored data point
        if E[i] == 1:
            # Add to uncensored index list
            uncensor_idx.append(i)
        else:
            # Add to censored index list
            censor_idx.append(i)

    # Separate y_true and y_pred
    y_pred_censor = y_pred[censor_idx]
    y_true_censor = y_true[censor_idx]
    y_pred_uncensor = y_pred[uncensor_idx]
    y_true_uncensor = y_true[uncensor_idx]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = - (torch.sum(torch.log(reduc_phi_censor)) + torch.sum(torch.log(reduc_phi_uncensor)) - torch.sum(torch.log(reduc_z_censor)) - torch.sum(torch.log(reduc_z_uncensor)))

    if reduction == 'mean':
        loss = loss / E.shape[0]

    return loss

# KL loss
# def _build_compile(self, model_input):
#         z_mean, z_log_var, z = self.encoder(model_input)
#         surved_y_output = self.decoder_y(z)
#         surved = Model(model_input, surved_y_output, name='SurVED')

#         kl_loss_orig = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
#         kl_loss = kl_loss_orig * self.kl_loss_weight
#         surved.add_loss(K.mean(kl_loss))
#         surved.add_metric(kl_loss_orig, name='kl_loss', aggregation='mean')
#         opt = Adam(lr=self.surved_lr)
#         surved.compile(loss=self._get_loss(), optimizer=opt, metrics=[self.cindex, self.surv_mse_loss])
#         return surved

def _get_loss(surv_mse_loss_weight, c_index_lb_weight):
        if surv_mse_loss_weight == 0:
            def _loss(y_true, y_pred):
                return surv_mse_loss(y_true, y_pred)*surv_mse_loss_weight
        else:
            def _loss(y_true, y_pred):
                return surv_mse_loss(y_true, y_pred)*surv_mse_loss_weight \
                       - cindex_lowerbound(y_true, y_pred)*c_index_lb_weight
        return _loss

def surv_mse_loss(self, y_true, y_pred):
        e = y_true[:, 1]
        y_diff = (y_true[:, 0] - y_pred[:, 0])
        err_func = torch.abs  #'abs'
        err = self.events_weight * e * err_func(y_diff) + self.censored_weight * (1 - e) * err_func(torch.relu(y_diff))
        return torch.mean(err)

def cindex_lowerbound(y_true, y_pred):
    y = y_true[:, 0]
    e = y_true[:, 1]
    ydiff = y.view(1, -1) - y.view(-1, 1)
    yij = torch.greater(ydiff, 0).float() + torch.equal(ydiff, 0).float() * \
        (e.view(-1, 1) != e.view(1, -1)).float()  # yi > yj
    is_valid_pair = yij * e.view(-1, 1)

    ypdiff = torch.transpose(y_pred) - y_pred  # y_pred[tf.newaxis,:] - y_pred[:,tf.newaxis]
    ypij = (1 + torch.log(torch.sigmoid(ypdiff))) / torch.log(2.0)
    cidx_lb = (torch.sum(ypij * is_valid_pair)) / torch.sum(is_valid_pair)
    return cidx_lb