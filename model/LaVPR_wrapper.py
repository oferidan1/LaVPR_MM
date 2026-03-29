from xml.parsers.expat import model
import torch
import numpy as np
#add parent directory to path
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import vpr_models
from model.LaVPR import LaVPR
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModel, AutoProcessor
import open_clip


class LaVPR_wrapper():
    def __init__(self, args):
        self.model_name = args.model_name
        self.vpr_model_name = args.vpr_model_name
        self.text_model_name = args.text_model_name
        self.device = args.device
        self.text_encoder_dim = args.text_dim
        self.vpr_encoder_dim = args.vpr_dim
            
        if args.is_dual_encoder or args.encode_mode!='both':    
            if 'bge' in self.text_model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)  
                self.text_encoder = AutoModel.from_pretrained(self.text_model_name).to(args.device)
                self.text_encoder.eval()
            else:
                self.text_encoder = SentenceTransformer(self.text_model_name, model_kwargs={"attn_implementation": "sdpa"}).to(args.device)    
            
            self.vpr_encoder = vpr_models.get_model(args.vpr_model_name.lower(), args.vpr_model_backbone, self.vpr_encoder_dim)
            self.vpr_encoder = self.vpr_encoder.eval().to(args.device)

            if args.bfloat16:
                self.vpr_encoder = self.vpr_encoder.bfloat16()
                self.text_encoder = self.text_encoder.bfloat16()
            
            self.encoder_dim = self.text_encoder_dim + self.vpr_encoder_dim
            if args.encode_mode == 'text':
                self.encoder_dim = self.text_encoder_dim 
            elif args.encode_mode == 'image':
                self.encoder_dim = self.vpr_encoder_dim
        else:            
            self.single_encoder = LaVPR(   
                #---- Encoder
                vpr_model_name=args.vpr_model_name.lower(),
                vpr_model_backbone=args.vpr_model_backbone,
                vpr_encoder_dim=args.vpr_dim,      
                fusion_type=args.fusion_type,
                is_encode_image=args.is_encode_image,
                is_encode_text=args.is_encode_text,        
                embeds_dim=args.embeds_dim,
                is_text_pooling=args.is_text_pooling,
                is_image_pooling=args.is_image_pooling,
                text_model_name=args.text_model_name,
                text_encoder_dim=args.text_dim,                
            )

            model_state_dict = torch.load(args.model_name)['state_dict']
            self.single_encoder.load_state_dict(model_state_dict)
            
            self.single_encoder = self.single_encoder.to(args.device)
            self.single_encoder.eval()            
             
            self.encoder_dim = self.single_encoder.embeds_dim           
            
        
    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Sum of the attention mask
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
        # Mean Pooling
        return sum_embeddings / sum_mask
            
    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vpr_encoder(images)
            text_features = self.encode_text(texts)       
        return image_features, text_features
    
    def encode_single(self, images, texts):
        with torch.no_grad():
            features, text_features, w, _, _ = self.single_encoder(images, texts)
        return features, text_features, w 
    
    def encode_image(self, images):
        with torch.no_grad():
                image_features = self.vpr_encoder(images)            
        return image_features
    
    def encode_text(self, texts):
        if 'bge' in self.text_model_name:                    
            text_tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():      
                model_output = self.text_encoder(**text_tokens)                        
                text_features = model_output[0][:, 0]            
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)   
        else:
            text_features = self.text_encoder.encode(texts, convert_to_tensor=True)
        return text_features


        
