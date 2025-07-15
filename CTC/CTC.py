import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
		
		Initialize instance variables

		Argument(s)
		-----------
		
		BLANK (int, optional): blank label index. Default 0.

		"""

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
		"""

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        skip_connect = []
        for i, symbol in enumerate(extended_symbols):
            if i + 2 < N and extended_symbols[i] != extended_symbols[i + 2]:
                skip_connect.append(1)
            else:
                skip_connect.append(0)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        # return extended_symbols, skip_connect
        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        alpha[0][0] = logits[0][extended_symbols[0]]
        # TODO: Intialize alpha[0][1]
        alpha[0][1] = logits[0][extended_symbols[1]]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        for t in range(1, T):
            for s in range(S):
                alpha[t][s] = alpha[t - 1][s]
                if s - 1 >= 0:
                    alpha[t][s] += alpha[t - 1][s - 1]
                if skip_connect[s] and s - 2 >= 0:
                    alpha[t][s] += alpha[t - 1][s - 2]

                alpha[t][s] *= logits[t][extended_symbols[s]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
		
		"""

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        beta[T-1][-1] = logits[T-1][extended_symbols[-1]]
        beta[T-1][-2] = logits[T-1][extended_symbols[-2]]

        for t in reversed(range(T-1)):
            for s in reversed(range(S)):
                beta[t][s] = beta[t+1][s]
                if s+1 < S:
                    beta[t][s] += beta[t+1][s+1]
                if skip_connect[s] and s+2 < S:
                    beta[t][s] += beta[t+1][s+2]
                beta[t][s] *= logits[t][extended_symbols[s]]

        for t in reversed(range(T)):
            for s in reversed(range(S)):
                beta[t][s] /= logits[t][extended_symbols[s]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

		"""

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        gamma = alpha * beta
        sumgamma = np.sum(gamma, axis=1)
        gamma /= sumgamma

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
        
		"""
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            target[batch_itr] = target[batch_itr, :target_lengths[batch_itr]]
            #     Truncate the logits to input length
            logits[:, batch_itr, :] = logits[:input_lengths[batch_itr], batch_itr, :]
            #     Extend target sequence with blank
            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target[batch_itr])
            self.extended_symbols.append(ext_symbols)
            #     Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits, ext_symbols, skip_connect)
            #     Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits, ext_symbols, skip_connect)
            #     Compute posteriors using total probability function
            posterior_prob = self.ctc.get_posterior_probs(alpha, beta)
            #     Compute expected divergence for each batch and store it in totalLoss
            for t in range(input_lengths[batch_itr]):
                for s in range(len(ext_symbols)):
                    posterior_prob[t][s] *= np.log(logits[t, batch_itr, ext_symbols[s]])
            total_loss[batch_itr] = np.sum(np.sum(posterior_prob, axis=1), axis=0) * -1

            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            pass

        total_loss = np.sum(total_loss) / B

        return total_loss

    def backward(self):
        """
		
		CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
		w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        target = self.target
        logits = self.logits
        input_lengths = self.input_lengths
        target_lengths = self.target_lengths

        for batch_itr in range(B):
            # -------------------------------------------->
            #     Truncate the target to target length
            target[batch_itr] = target[batch_itr, :target_lengths[batch_itr]]
            #     Truncate the logits to input length
            logits[:, batch_itr, :] = logits[:input_lengths[batch_itr], batch_itr, :]
            #     Extend target sequence with blank
            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target[batch_itr])

            #     Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits, ext_symbols, skip_connect)
            #     Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits, ext_symbols, skip_connect)
            #     Compute posteriors using total probability function
            posterior_prob = self.ctc.get_posterior_probs(alpha, beta)
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            posterior_prob = self.ctc.get_posterior_probs(alpha, beta)
            for t in range(T):
                dY[t, batch_itr, :] = np.zeros(C)
                for s in range(len(ext_symbols)):
                    dY[t, batch_itr, ext_symbols[s]] -= (posterior_prob[t][s] / logits[t, batch_itr, ext_symbols[s]])
            pass

        return dY

