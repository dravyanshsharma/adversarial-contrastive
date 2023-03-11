import torch
from torch.nn import functional as F
import apex.amp as amp



class Attacker(object):
    def __init__(self, epsilon, step_size, num_steps, loss, contrast, rand_init=True):
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.loss_fn = loss_dict[loss]
        self.contrast = contrast
        self.rand_init = rand_init
    
    def __call__(self, x_nat, model, optimizer, args):
        model.eval()

        if self.rand_init:
            x_adv = x_nat.detach() + 0.001 * torch.randn_like(x_nat)
        else:
            x_adv = x_nat.detach()
        
        z_nat = model(x_nat) #.detach()
        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = self.loss_fn(model, x_adv, z_nat, self.contrast)
            if args.amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    grad = torch.autograd.grad(scaled_loss, [x_adv])[0].detach()
            else:
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            optimizer.zero_grad()
            
        return x_adv


def l2_loss_inter(model, x_adv, z_nat, contrast):
    '''
    L2 loss between two independent crops
    '''
    x, y = torch.chunk(model(x_adv), 2, dim=0)
    return (x - y).pow(2).sum(dim=-1).pow(0.5).mean()


def l2_loss_intra(model, x_adv, z_nat, contrast):
    '''
    L2 loss between natural and adversarial versions of the same crop
    '''
    z_adv = model(x_adv)
    return (z_nat - z_adv).pow(2).sum(dim=-1).pow(0.5).mean()


def kl_loss(model, x_adv, z_nat, contrast):
    prob_nat = F.softmax(contrast(z_nat), dim=1)
    z_adv = model(x_adv)
    log_prob_adv = F.log_softmax(contrast(z_adv), dim=1)
    return F.kl_div(log_prob_adv, prob_nat, reduction='sum')


def ntce_loss(model, x_adv, z_nat, contrast):
    logits = contrast(model(x_adv))
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)



loss_dict = {
    'l2_inter': l2_loss_inter,
    'l2_intra': l2_loss_intra,
    'kl': kl_loss,
    'ntce': ntce_loss,
}
