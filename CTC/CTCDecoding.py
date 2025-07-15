import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        S, T, B = y_probs.shape

        # TODO:
        symbols = self.symbol_set.copy()
        symbols.insert(0, '-')
        idx = np.argmax(y_probs, axis=0)
        path = []
        for t in range(T):
            path_prob *= y_probs[idx[t], t, 0]
            path.append(symbols[idx[t]])

        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        t = 0
        while t < T:
            j = t
            while j < T and symbols[j] == symbols[t]:
                j += 1
            if symbols[t] != '-':
                decoded_path.append(symbols[t])
            t = j

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        self.y_probs = y_probs
        S, T, B = y_probs.shape
        # T = y_probs.shape[1]
        # #bestPath, FinalPathScore = None, None
        # TempBestPaths = {}
        # BestPaths = {'-' : 1.0}
        # for t in range(T):
        #     symbol_probs = y_probs[:, t, 0]
        #     for path, score in BestPaths[:self.beam_width].items():
        #         for i, symbol_prob in enumerate(symbol_probs):
        #             if self.symbol_set[i] == path[-1]:
        #
        #             elif path[-1] == '-':
        #                 path[-1] = self.symbol_set[i]
        #             elif self.symbol_set[i] != '-':
        #                 path += self.symbol_set[i]
        #             TempBestPaths[path] = score * symbol_prob
        #         BestPaths = TempBestPaths
        #     TempBestPaths.clear()
        #
        # for path, score in BestPaths.items():
        #     if path[-1] == '-':
        #         path = path[: -1]
        # MergedPathScores = {}
        # BestPath = '-'
        # BestScore = 0
        # for path, score in BestPaths.items():
        #     MergedPathScores[path] = score
        #     if score > BestScore:
        #         BestPath = path
        #         BestScore = score
        #
        # return BestPath, BestScore

        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol = self.InitializePaths()

        # Subsequent time steps
        for t in range(1, T):
            # Prune the collection down to the BeamWidth
            PathsWithTerminalBlank, PathsWithTerminalSymbol = self.Prune(NewPathsWithTerminalBlank,
                                                                         NewPathsWithTerminalSymbol)

            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank,
                                                                                PathsWithTerminalSymbol,
                                                                                t)

            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank,
                                                                             PathsWithTerminalSymbol,
                                                                             t)

        # Merge identical paths differing only by the final blank
        MergedPaths = self.MergeIdenticalPaths(NewPathsWithTerminalBlank,
                                               NewPathsWithTerminalSymbol)

        # Pick best path
        BestPath = sorted(MergedPaths, key=lambda item: item[1], reverse=True)[0]  # Find the path with the best score

        return BestPath, MergedPaths

    def InitializePaths(self):

        InitialPathsWithTerminalBlank = {'': self.y_probs[0, 0, 0]}

        InitialPathsWithTerminalSymbol = {}
        for i, c in enumerate(self.symbol_set):
            path = c
            InitialPathsWithTerminalSymbol[path] = self.y_probs[i, 0, 0]

        return InitialPathsWithTerminalBlank, InitialPathsWithTerminalSymbol

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, t):
        UpdatedPathsWithTerminalBlank = {}

        for path, score in PathsWithTerminalBlank.items():
            UpdatedPathsWithTerminalBlank[path] = score * self.y_probs[0, t, 0]

        for path, score in PathsWithTerminalSymbol.items():
            if path in UpdatedPathsWithTerminalBlank.keys():
                UpdatedPathsWithTerminalBlank[path] += score * self.y_probs[0, t, 0]
            else:
                UpdatedPathsWithTerminalBlank[path] = score * self.y_probs[0, t, 0]

        return UpdatedPathsWithTerminalBlank

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, t):
        UpdatedPathsWithTerminalSymbol = {}

        for path, score in PathsWithTerminalBlank.items():
            for i, c in enumerate(self.symbol_set):
                new_path = path + c
                UpdatedPathsWithTerminalSymbol[new_path] = score * self.y_probs[i, t, 0]

        for path, score in PathsWithTerminalSymbol.items():
            for i, c in enumerate(self.symbol_set):
                if c == path[-1]:
                    new_path = path
                else:
                    new_path = path + c

                if new_path in UpdatedPathsWithTerminalSymbol.keys():
                    UpdatedPathsWithTerminalSymbol[new_path] += score * self.y_probs[i, t, 0]
                else:
                    UpdatedPathsWithTerminalSymbol[new_path] = score * self.y_probs[i, t, 0]

        return UpdatedPathsWithTerminalSymbol

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol):

        PathsWithTerminalBlank = dict(sorted(PathsWithTerminalBlank.items(), key=lambda item: item[1], reverse=True))
        PrunedPathsWithTerminalBlank = {}
        for i, path, score in enumerate(PathsWithTerminalBlank.items()):
            if i < self.beam_width:
                PrunedPathsWithTerminalBlank[path] = score
            else:
                break

        PathsWithTerminalSymbol = dict(sorted(PathsWithTerminalSymbol.items(), key=lambda item: item[1], reverse=True))
        PrunedPathsWithTerminalSymbol = {}
        for i, path, score in enumerate(PathsWithTerminalSymbol.items()):
            if i < self.beam_width:
                PrunedPathsWithTerminalSymbol[path] = score
            else:
                break

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, PathsWithTerminalSymbol):

        MergedPaths = PathsWithTerminalSymbol

        for path, score in PathsWithTerminalBlank:
            if path in MergedPaths.keys():
                MergedPaths[path] += score
            else:
                MergedPaths[path] = score

        return MergedPaths
