"this is the implementation of muon optimizer"


def newton_schulz_inverse_sqrt(A, iteration= 5, eps= 1e-6):   

    dim = A.shape[0]
    I = torch.eye(dim, device=device, dtype=A.dtype)  
    normA =  A.norm()
    Y  = A / A.norm()  
    Z  = I.clone()

    for _ in range(iteration):

         T = 0.5 * (3 * I - Z @ Y)   
         Y = Y @ T
         Z = T @ Z  
    
    return Z / torch.sqrt(normA + eps)  



class Muon(torch.optim.Optimizer):   

      def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, ns_iters=3, eps=1e-8):   
          
          defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_iters=ns_iters, eps=eps)
          super().__init__(params, defaults)

      @torch.no_grad()
      def step(self, closure = None):  

          loss = None 
          if closure is not None:  
               with torch.enable_grad():
                loss = closure()  

          for group in self.param_groups:   
            
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_iters = group['ns_iters']
            eps = group['eps']    

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
            
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                m = state['momentum_buffer'] 
                m.mul_(momentum).add_(g, alpha=(1 - momentum))  

                W = p.data.view(p.shape[0], -1)
                G = g.view_as(W)  
                A = G @ G.T / (G.norm() + eps)   
                A_inv_sqrt = newton_schulz_inverse_sqrt(A, iteration=ns_iters, eps=eps)  
                G_tilde = (A_inv_sqrt @ G).view_as(g)     

                param_norm = W.norm()
                grad_norm = G_tilde.norm() + eps  
                scale = param_norm / grad_norm  

                update = lr* scale* m 
                p.add_(-update)
                p.add_(-lr * weight_decay * p)  
                state['momentum_buffer'] = m
          
          return loss 

model = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
).cuda()

optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.9)  
x = torch.randn(32, 256, device='cuda')
y = torch.randint(0, 10, (32,), device='cuda')
criterion = torch.nn.CrossEntropyLoss()
   
for _ in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()  
    print(loss)
