import torch
import torch.nn as nn

class ADDLoss(nn.Module):
    def __init__(self):
        super(ADDLoss, self).__init__()

    def forward(self, pred_R, pred_T, gt_R, gt_T, model_points):
        """
        pred_R: (B, 9) -> Rappresentazione della matrice 3x3 appiattita
        pred_T: (B, 3) -> Traslazione 3D predetta
        gt_R: (B, 3, 3) -> Matrice di rotazione Ground Truth
        gt_T: (B, 3) -> Traslazione Ground Truth
        model_points: (B, N, 3) -> Punti 3D del modello CAD
        """
        # 1. Reshape della rotazione predetta (i 9 neuroni richiesti)
        pred_R = pred_R.view(-1, 3, 3)
        
        # 2. Trasformazione punti con posa predetta: R*p + T
        # model_points: (B, N, 3), pred_R: (B, 3, 3)
        # Usiamo bmm per moltiplicazione tra batch e aggiungiamo T con unsqueeze
        pred_points = torch.bmm(model_points, pred_R.transpose(1, 2)) + pred_T.unsqueeze(1)
        
        # 3. Trasformazione punti con posa GT: R_gt*p + T_gt
        gt_points = torch.bmm(model_points, gt_R.transpose(1, 2)) + gt_T.unsqueeze(1)

        # --- ADD STANDARD (Oggetti asimmetrici) ---
        # Distanza media tra punti corrispondenti nello spazio
        loss = torch.mean(torch.norm(pred_points - gt_points, dim=2))
            
        return loss