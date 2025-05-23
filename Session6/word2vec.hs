{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString as BS (concat, pack, unpack, fromStrict, toStrict, length)
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M (member ,insertWith,findWithDefault, fromList, toList, Map, lookup, empty) -- add containers to dependencies in package.yaml
import Data.List (nub)
import Data.Word8 as W (toLower)
import Data.Typeable
import Data.Maybe (fromMaybe)
import Data.Char (isAsciiLower, isAsciiUpper)

import System.Random (randomRIO)
import Control.Monad (when)

import qualified Torch.Functional.Internal as FI (log)
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', mean)
import Torch.NN (Parameterized(..), Parameter, Linear)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye', zeros')
import Torch.Optim as Opt
import Torch.Functional (embedding', mean, logSoftmax, nllLoss', Dim(..), matmul, transpose2D)
import ML.Exp.Chart (drawLearningCurve)
import Torch.TensorFactories (eye', zeros',randIO')
import Torch.Random (randn')

-- your text data (try small data first)
textFilePath = "data/sampleFinal.txt"
modelPath =  "data/test.params"
wordLstPath = "data/sample_wordlst.txt"


data EmbeddingSpec = EmbeddingSpec 
                    {
                        wordNum :: Int, -- the number of words
                        wordDim :: Int  -- the dimention of word embeddings
                    } deriving (Show, Eq, Generic)

data Embedding = Embedding 
                {
                    wordEmbedding :: Parameter
                } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model 
            {
                embeddings :: Embedding,
                outputWeights :: Parameter
            } deriving (Generic, Parameterized)

batch :: Int
batch = 128

epoch :: Int
epoch = 100

learningRate :: Float
learningRate = 0.001

iteration :: Int
iteration = 5

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar w =
    let c = toEnum (fromEnum w) :: Char
    in not (isAsciiLower c || isAsciiUpper c || c == ' ' )

preprocess ::
    B.ByteString -> -- input
    [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
    where
        filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
        textLines = B.split (head $ encode "\n") filteredtexts




lossCE :: Tensor -> Tensor -> Tensor
lossCE yTrue yPred =
    nllLoss' (logSoftmax (Dim 1) yPred) yTrue

wordTuples :: (B.ByteString -> Int) -> [B.ByteString] -> [(Int, Int)]
wordTuples wordToIndex wordlst =
    concat [createTupleFromCentral (wordToIndex (wordlst !! i)) (map wordToIndex (getContext i)) | i <- [2 .. length wordlst - 3]]
    where
        getContext i =
            [wordlst !! j | j <- [i - 2 .. i + 2], j /= i, j >= 0, j < length wordlst]

createTupleFromCentral :: Int -> [Int] -> [(Int, Int)]
createTupleFromCentral central t = 
    [(central, nbt) | nbt <- t]

wordFreq :: [B.ByteString] -> M.Map B.ByteString Int
wordFreq [] = M.empty
wordFreq (x:xs) =
    let mapFreq = wordFreq xs
    in M.insertWith (+) x 1 mapFreq

tensorInputWords :: [(Int, Int)] -> Tensor
tensorInputWords pairs = 
    asTensor ([central | (central, _) <- pairs])

tensorTargetWords :: [(Int, Int)] -> Tensor
tensorTargetWords pairs =
    asTensor ([context | (_, context) <- pairs])

shuffle :: [Int] -> [(Int, Int)] -> [(Int, Int)]
shuffle [] pairs = pairs
shuffle (i:is) pairs = 
    let (firsts, rest) = splitAt (i `mod` length pairs) pairs
    in (head rest) : shuffle is (firsts ++ tail rest)

makeBatches :: [(Int, Int)] -> [[(Int, Int)]]
makeBatches [] = []
makeBatches pairs 
    | length pairs <= batch = [pairs]
    | otherwise = 
        let taken = take batch pairs
            remaining = drop batch pairs
        in taken : makeBatches remaining

train :: EmbeddingSpec -> Model -> [Int] -> [[(Int, Int)]] -> IO Model
train embeddingSpec model indexWord batches = do
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
    (modelFinal, _, lossValues) <- foldLoop (model, optimizer, []) iteration $ \(currentModel, currOptimizer, losses) epoch -> do
        putStrLn ("Epoch " ++ show epoch)
        (modelAfterEpoch, bOptimizer, lossesInEpoch) <- foldLoop (currentModel, currOptimizer, []) (length batches) $ \(currentBatchModel, bOptimizer, batchLosses) i -> do
            let currBatch = batches !! (i - 1)
                input = tensorInputWords currBatch
                target = tensorTargetWords currBatch
                embBatch = embedding' (toDependent $ wordEmbedding $ embeddings currentBatchModel) input
                logits = matmul embBatch (transpose2D (toDependent $ outputWeights currentBatchModel))
                logProbs = logSoftmax (Dim 1) logits
                lossTensor = nllLoss' target logProbs
                lossValue = asValue lossTensor :: Float
            (model', optimizer') <- runStep currentBatchModel bOptimizer lossTensor (asTensor learningRate) :: IO (Model, Adam)
            when (i `mod` 50 == 0) $ do
                putStrLn ("Model save for batch : " ++ show i)
                saveParams model' modelPath
                drawLearningCurve ("curve4/curve_batch" ++ show i ++ ".png") "Learning Curve" [("Training Loss", reverse (lossValue : batchLosses ++ losses))]
            return (model', optimizer', lossValue : batchLosses)
        return (modelAfterEpoch, bOptimizer, lossesInEpoch ++ losses)
    drawLearningCurve "curve4/FINALcurveTrain.png" "Learning Curve" [("Training Loss", reverse lossValues)]
    return modelFinal

--embTxt = embedding' (toDependent $ wordEmbedding emb) (asTensor idxes)

subsamplingFreq :: [(B.ByteString, Int)] -> [B.ByteString] -> [(B.ByteString, Float)]
subsamplingFreq [] _ = []
subsamplingFreq (x:xs) textSample = 
    let wordProb = subsamplingFreq xs textSample
        (word, nb) = x
        f = fromIntegral nb / fromIntegral (length textSample)
        p = max 0 (sqrt(1e-5 / f))
    in (word, realToFrac p) : wordProb

keepOrNot :: M.Map B.ByteString Float -> [B.ByteString] -> IO [B.ByteString]
keepOrNot _ [] = do
    return []
keepOrNot probWord (x:xs) = do
    keeping <- keepOrNot probWord xs
    r <- randomRIO (0.0, 1.0)
    let p = fromMaybe 0.00 (M.lookup x probWord)
    if r > p
        then return (x : keeping)
    else
        return (keeping)

main :: IO ()
main = do
    -- load text file
    texts <- B.readFile textFilePath

    -- Create a unique word list
    let wordLines = preprocess texts 
        textSample =  map BS.fromStrict $ map BS.pack $ map (map W.toLower) $ map BS.unpack $ map BS.toStrict $ concat wordLines
        -- wordlst = nub $ BS.fromStrict $ BS.pack $ map W.toLower $ BS.unpack $ BS.toStrict $ concat wordLines
        wordlst = nub textSample
        freq = M.toList $ wordFreq textSample
        probTable = subsamplingFreq freq textSample
    putStrLn "probTable finished"
    textSampleProb <- keepOrNot (M.fromList probTable) textSample
    let newWordlst = nub textSampleProb
        wordToIdxMap = M.fromList (zip newWordlst [0..])
        lookupIndex w = fromMaybe 0 (M.lookup w wordToIdxMap)
        wordToIndex = lookupIndex
        filteredTextSampleProb = filter (\w -> M.member w wordToIdxMap) textSampleProb
        indexWord = map wordToIndex newWordlst
        wordWithIndex = zip newWordlst indexWord
    putStrLn "WordWithIndex finished"
    let trainingPairByte = wordTuples wordToIndex filteredTextSampleProb
        trainingPairByteshuffled = shuffle [324] trainingPairByte

    -- Create initial embedding (wordDim × wordNum)
    let embeddingSpec = EmbeddingSpec {wordNum = length newWordlst + 1 , wordDim = 30}
    outputWeights <- makeIndependent $ zeros' [wordNum embeddingSpec, wordDim embeddingSpec]
    wordEmb <- makeIndependent =<< randIO' [wordNum embeddingSpec, wordDim embeddingSpec]
    let emb = Embedding { wordEmbedding = wordEmb }

    -- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.
    model <- return $ Model emb outputWeights
    -- Save params to use trained parameter in the next session
    -- trainedEmb :: Embedding
    let batches = makeBatches trainingPairByteshuffled
    putStrLn ("length batches : " ++ show (length batches))
    trainedModel <- train embeddingSpec model indexWord batches

    saveParams trainedModel modelPath
    -- Save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") newWordlst)
    
    -- Load params
    -- initWordEmb <- makeIndependent $ zeros' [1]
    -- let initEmb = Embedding {wordEmbedding = initWordEmb}
    -- loadedEmb <- loadParams initEmb modelPath

    return ()