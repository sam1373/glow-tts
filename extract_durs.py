import numpy as np
from Bio import pairwise2

from text.symbols import symbols

import torch
from torch.nn import functional as F

def interleave(x, y):
    xy = torch.stack([x[-1], y], dim=1).view(-1)
    xy = F.pad(xy, pad=[0, 1], value=x[-1])
    return xy


class PadProcesser:
    def __init__(self):
        labels = symbols# + ['~']
        self.blank_id = 0#len(labels) - 1
        self.space_id = labels.index(' ')
        self.labels_map = dict([(i, labels[i]) for i in range(len(labels))])

    def bound_text(self, tokens):
        return [self.space_id] + tokens + [self.space_id]

    def bound_ctc(self, tokens, logprobs):
        tokens = [self.space_id, self.blank_id] + tokens + [self.blank_id, self.space_id]

        logprobs = np.lib.pad(logprobs, ((2, 2), (0, 0)), 'edge')

        def swap(col, a, b):
            logprobs[col][a], logprobs[col][b] = logprobs[col][b], logprobs[col][a]

        first_token, last_token = tokens[2], tokens[-3]
        swap(0, first_token, self.space_id)
        swap(1, first_token, self.blank_id)
        swap(-1, last_token, self.space_id)
        swap(-2, last_token, self.blank_id)

        return tokens, logprobs

    def merge(self, tokens):
        output_tokens = []
        output_cnts = []
        cnt = 0
        for i in range(len(tokens)):
            if i != 0 and (tokens[i - 1] != tokens[i]):
                output_tokens.append(tokens[i - 1])
                output_cnts.append(cnt)

                cnt = 0

            cnt += 1

        output_tokens.append(tokens[-1])
        output_cnts.append(cnt)

        assert sum(output_cnts) == len(tokens), f'SUM_CHECK {sum(output_cnts)} vs {len(tokens)}'

        return output_tokens, output_cnts

    def merge_with_blanks(self, tokens, cnts, logprobs=None):
        def choose_sep(l, r, a, b):
            # `tokens[l] == a and tokens[r] == b`.
            sum_a, sum_b = logprobs[l, a], logprobs[l + 1:r + 1, b].sum()
            best_sum, best_sep = sum_a + sum_b, 0
            for sep in range(1, r - l):
                sum_a += logprobs[l + sep, a]
                sum_b -= logprobs[l + sep, b]
                if sum_a + sum_b > best_sum:
                    best_sum, best_sep = sum_a + sum_b, sep

            return best_sep

        output_tokens = []
        output_durs = []
        blank_cnt = 0
        total_cnt = 0
        for token, cnt in zip(tokens, cnts):
            total_cnt += cnt
            if token == self.blank_id:
                blank_cnt += cnt
                continue

            output_tokens.append(token)

            if logprobs is None:
                # Half half.
                left_cnt = blank_cnt // 2
            else:
                # Clever sep choice based on sum of log probs.
                left_cnt = choose_sep(
                    l=total_cnt - cnt - blank_cnt - 1,
                    r=total_cnt - cnt,
                    a=output_tokens[-1],
                    b=token,
                )
            right_cnt = blank_cnt - left_cnt
            blank_cnt = 0

            if left_cnt:
                output_durs[-1] += left_cnt
            output_durs.append(cnt + right_cnt)

        output_durs[-1] += blank_cnt

        assert sum(output_durs) == sum(cnts), f'SUM_CHECK {sum(output_durs)} vs {sum(cnts)}'

        return output_tokens, output_durs

    def align(self, output_tokens, gt_text):
        def make_str(tokens):
            return ''.join(self.labels_map[c] for c in tokens)

        s = make_str(output_tokens)
        t = make_str(gt_text)
        alignmet = pairwise2.align.globalxx(s, t, gap_char='%')[0]
        sa, ta, *_ = alignmet
        return sa, ta

    def generate(self, gt_text, alignment, durs):
        output_tokens = []
        output_cnts = []
        si, ti = 0, 0
        #         print(len(durs))
        assert len(alignment[0]) == len(alignment[1])
        for sc, tc in zip(*alignment):
            #             print(si, sc, ti, tc)
            if sc == '%' and tc == '%':
                print('NO WAY')
                continue

            if sc == '%':
                output_tokens.append(self.blank_id)
                output_cnts.append(durs[si])
                si += 1
            elif tc == '%':
                output_tokens.append(gt_text[ti])
                output_cnts.append(0)
                ti += 1
            else:
                output_tokens.append(gt_text[ti])
                output_cnts.append(durs[si])
                si += 1
                ti += 1

        assert sum(output_cnts) == sum(durs)

        return output_tokens, output_cnts

    def __call__(self, text, ctc_tokens, ctc_logprobs, mel_len):
        # This adds +2 tokens.
        text = self.bound_text(text)
        # This add +4 tokens, 2 of them are blank.
        ctc_tokens, ctc_logprobs = self.bound_ctc(ctc_tokens, ctc_logprobs)

        ctc_tokens, ctc_cnts = self.merge(ctc_tokens)
        ctc_tokens, ctc_durs = self.merge_with_blanks(ctc_tokens, ctc_cnts, ctc_logprobs)

        alignment = self.align(text, ctc_tokens)
        tokens, cnts = self.generate(text, alignment, ctc_durs)
        tokens, durs = self.merge_with_blanks(tokens, cnts)
        assert tokens == text, 'EXACT_TOKENS_MATCH_CHECK'

        def adjust(start, direction, value):
            i = start
            while value != 0:
                dur = durs[i]

                if value < 0:
                    durs[i] = dur - value
                else:
                    durs[i] = max(dur - value, 0)

                value -= dur - durs[i]
                i += direction

        adjust(0, 1, 4)
        adjust(-1, -1, sum(durs) - mel_len)  # Including 4 suffix bound tokens.
        assert durs[0] >= 0, f'{durs[0]}'
        assert durs[-1] >= 0, f'{durs[-1]}'

        durs = np.array(durs, dtype=np.long)
        assert durs.shape[0] == len(text), f'LEN_CHECK {durs.shape[0]} vs {len(text)}'
        assert np.sum(durs) == mel_len, f'SUM_CHECK {np.sum(durs)} vs {mel_len}'

        return durs


class Seq:
    def __init__(self, tokens, cnts=None):
        if cnts is None:
            cnts = np.ones(len(tokens), dtype=np.long)

        assert len(tokens) == len(cnts)
        self.tokens = tokens
        self.cnts = cnts

    def __repr__(self):
        return repr(list(zip(self.tokens, self.cnts)))

    @property
    def total(self):
        return sum(self.cnts)

    def merge(self):
        output_tokens = []
        output_cnts = []

        cnt = 0
        for i in range(len(self.tokens)):
            if i != 0 and (self.tokens[i - 1] != self.tokens[i]):
                output_tokens.append(self.tokens[i - 1])
                output_cnts.append(cnt)

                cnt = 0

            cnt += self.cnts[i]

        output_tokens.append(self.tokens[-1])
        output_cnts.append(cnt)

        assert sum(output_cnts) == sum(self.cnts), \
            f'SUM-CHECK {sum(output_cnts)} vs {sum(self.cnts)}'

        return Seq(output_tokens, output_cnts)

    def full_pad(self, blank_id, blank_cnt=1):
        output_tokens = [blank_id]
        output_cnts = [blank_cnt]

        for token, cnt in zip(self.tokens, self.cnts):
            output_tokens.append(token)
            output_cnts.append(cnt)

            output_tokens.append(blank_id)
            output_cnts.append(blank_cnt)

        return Seq(output_tokens, output_cnts)

    def adjust_cnt(self, value, start=-1, direction='left'):
        tokens, cnts = self.tokens, self.cnts.copy()

        i, di = start, -1 if direction == 'left' else 1
        while value != 0:
            cnt = cnts[i]

            if value < 0:
                cnts[i] = cnt - value
            else:
                cnts[i] = max(cnt - value, 0)

            value -= cnt - cnts[i]
            i += di

        return Seq(tokens, cnts)

    def split2(self):
        tokens1, cnts1 = [], []
        tokens2, cnts2 = [], []
        turn = 1

        for token, cnt in zip(self.tokens, self.cnts):
            if turn == 1:
                tokens1.append(token)
                cnts1.append(cnt)
            else:
                tokens2.append(token)
                cnts2.append(cnt)

            turn = 1 if turn == 2 else 2

        return Seq(tokens1, cnts1), Seq(tokens2, cnts2)


class DurationExtractor(PadProcesser):
    def __call__(self, text, ctc_tokens, ctc_logprobs, mel_len):

        text = Seq(text).full_pad(self.blank_id)
        ctc = Seq(ctc_tokens).merge().full_pad(self.blank_id, blank_cnt=0).merge()

        alignment = self.align(text.tokens, ctc.tokens)
        gen = Seq(*self.generate(text.tokens, alignment, ctc.cnts)).merge()
        #         gen = gen.merge().adjust_cnt(gen.total - mel_len)

        print(gen.total, mel_len)
        # Two durs conditions.
        assert gen.tokens == text.tokens
        #         assert gen.total == mel_len
        #assert abs(2 * gen.total - mel_len) <= 1

        blanks, text = gen.split2()
        blanks = np.array(blanks.cnts, dtype=np.long)
        cnts = np.array(text.cnts, dtype=np.long)

        assert len(blanks) == len(cnts) + 1


        return blanks, cnts

