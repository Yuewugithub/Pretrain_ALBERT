import os
import pandas as pd
import torch
from transformers import AlbertForMaskedLM, AlbertTokenizer, AutoTokenizer
from tqdm import tqdm


class Predictor(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self):
        print('loading tokenizer config ...')
        self.tokenizer = AlbertTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)

    def load_model(self):
        print('loading model...%s' % self.config.path_model_predict)
        self.model = AlbertForMaskedLM.from_pretrained(self.config.path_model_predict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, test_loader):
        print('predict start')
        progress_bar = tqdm(total=len(test_loader), desc='Predict')
        src = []
        label = []
        pred = []
        input = []
        total = 0
        count = 0

        for i, batch in enumerate(test_loader):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                outputs_pred = outputs.logits

            tmp_src = batch['input_ids'].cpu().numpy()
            tmp_label = batch['labels'].cpu().numpy()
            tmp_pred = torch.max(outputs_pred, -1)[1].cpu().numpy()

            for i in range(len(tmp_label)):
                line_s = tmp_src[i]
                line_l = tmp_label[i]
                line_l_split = [x for x in line_l if x not in [0]]
                line_p = tmp_pred[i]
                line_p_split = line_p[:len(line_l_split)]
                tmp_s = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_s))
                tmp_lab = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_l_split))
                tmp_p = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_p_split))
                input.append(tmp_s)
                label.append(tmp_lab)
                pred.append(tmp_p)

                if '[MASK]' in tmp_s:
                    total += 1
                    if tmp_lab == tmp_p:
                        count += 1

            progress_bar.update(1)

        progress_bar.close()

        acc = count / max(1, total)
        print('\nTask: acc=', acc)

        # 保存结果
        data = {'src': label, 'pred': pred, 'mask': input}
        data = pd.DataFrame(data)
        path = os.path.join(self.config.path_datasets, 'output')
        if not os.path.exists(path):
            os.mkdir(path)
        path_output = os.path.join(path, 'pred_data.csv')
        data.to_csv(path_output, sep='\t', index=False)
        print('Task 1: predict result save: {}'.format(path_output))