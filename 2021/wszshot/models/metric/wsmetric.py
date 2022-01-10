import torch
from torch.nn import CosineSimilarity
import pdb

class WSMetric:
    def __init__(self,
                 word_feature_map,
                 wordLabelEncoder):
        self.word_feature_map = {}
        for column in word_feature_map:
            #pdb.set_trace()
            self.word_feature_map[column] = torch.tensor(word_feature_map[column]).cuda()
            
        self.cosine = CosineSimilarity(dim=1, eps=1e-6)
        self.wordLabelEncoder = wordLabelEncoder
    
    def compute(self, pred, target,
                mode='similarity'):
        assert mode == 'similarity' or mode == 'accuracy', f"Error: Unknown mode {mode}"
        
        sim_stats = self._similarity(pred, target)
        acc_stats  = self._accuracy(pred, target)
        
        dict_stats = {}
        for key, value in sim_stats.items():
            dict_stats[f'similarity_{key}'] = value
        for key, value in acc_stats.items():
            dict_stats[f'accuracy_{key}'] = value
        
        #print(dict_stats)
        return dict_stats
    
    def _similarity(self, pred, target):
        # 'phos' feature similarity
        phos = torch.mean(self.cosine(pred['phos'], target['phos'])).item()
        # 'phoc' feature similarity
        phoc = torch.mean(self.cosine(pred['phoc'], target['phoc'])).item()
        # 'phosc' feature similarity
        _pred   = torch.cat((pred['phos'], pred['phoc']), 1)
        _target = torch.cat((target['phos'], target['phoc']), 1)
        phosc = torch.mean(self.cosine(_pred, _target)).item()
        
        return {'phos': phos, 'phoc': phoc, 'phosc': phosc}        
            
    def _accuracy(self, pred, target):
        phos_feat = self.word_feature_map['phos'] #torch.tensor(self.word_feature_map.loc[:, 'phos'])
        phoc_feat = self.word_feature_map['phoc'] #torch.tensor(self.word_feature_map.loc[:, 'phoc'])
        word_vec  = self.word_feature_map['word'] #torch.tensor(self.word_feature_map.loc[:, 'word'])
        
        #pdb.set_trace()
        
        n_words = word_vec.shape[0]
        # 'phoc' feature accuracy
        t_pred = pred['phos']
        indices = []
        for i in range(t_pred.shape[0]):
            pred_vec = t_pred[i, :].repeat(n_words, 1)
            sim = self.cosine(pred_vec, phos_feat)
            indices.append(torch.argmax(sim).item())
        phos_wvec = word_vec[indices]
        phos = torch.mean(torch.tensor(phos_wvec == target['wlabel'], dtype=torch.float)).item()
        
        # 'phoc' feature accuracy
        t_pred = pred['phoc']
        indices = []
        for i in range(t_pred.shape[0]):
            pred_vec = t_pred[i, :].repeat(n_words, 1)
            sim = self.cosine(pred_vec, phoc_feat)
            indices.append(torch.argmax(sim).item())
        phoc_wvec = word_vec[indices]
        phoc = torch.mean(torch.tensor(phoc_wvec == target['wlabel'], dtype=torch.float)).item()
        
        # 'phosc' feature accuracy
        t_pred = torch.cat((pred['phos'], pred['phoc']), 1)
        phosc_feat = torch.cat((phos_feat, phoc_feat), 1)
        indices = []
        for i in range(t_pred.shape[0]):
            pred_vec = t_pred[i, :].repeat(n_words, 1)
            sim = self.cosine(pred_vec, phosc_feat)
            indices.append(torch.argmax(sim).item())
        phosc_wvec = word_vec[indices]
        phosc = torch.mean(torch.tensor(phosc_wvec == target['wlabel'], dtype=torch.float)).item()
        phosc_word = self.wordLabelEncoder.inverse_transform(phosc_wvec.tolist())
        
        return {'phos': phos, 'phoc': phoc, 'phosc': phosc, 'word':phosc_word}
