module Evaluation (
    precision,
    recall,
    accuracy,
    confusionMatrix,
    f1score,
    microF1score,
    macroF1score,
    weightedF1score
) where


import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Tensor (size)
import Torch.Functional.Internal as FI (div)
import Torch.Functional (add)

precision :: Tensor -> Tensor -> Tensor
precision tp fp =
    tp `FI.div` (fp `add` tp)

recall :: Tensor -> Tensor -> Tensor
recall tp fn = 
    tp `FI.div` (tp `add` fn)

accuracy :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor 
accuracy tp tn fp fn = 
    let sTP = fromIntegral (size 0 tp)
        sFP = fromIntegral (size 0 fp)
        sTN = fromIntegral (size 0 tn)
        sFN = fromIntegral (size 0 fn)
    in (sTP + sTN) / (sTP + sFN + sFP + sTN) 

confusionMatrix :: Tensor -> Tensor -> Tensor -> Tensor -> [[Double]]
confusionMatrix tp tn fp fn = 
    [[asValue tp :: Double, asValue fp :: Double], 
     [asValue fn :: Double, asValue tn :: Double]]

f1score :: Double -> Double -> Double
f1score p r = 2 * ((p*r)/(p+r))

microF1score :: [Double] -> [Double] -> [Double] -> Double
microF1score tp fp fn =
    let sumTP = sum tp
        sumFP = sum fp
        sumFN = sum fn
    in sumTP / (sumTP + 0.5 * (sumFP + sumFN))

macroF1score :: [Double] -> Double
macroF1score sF1 = (sum sF1) / (fromIntegral (length sF1)) :: Double

weightedF1score :: [Double] -> [Double] -> Double
weightedF1score sF1 weights = 
    let weightedSum = sum $ zipWith (*) sF1 weights
        totalWeight = sum weights
    in weightedSum / totalWeight

-- main :: IO ()
-- main = do
--     let pre = precision tp fp
--         re = recall tp fn
--         accu = accuracy tp tn fp fn
--         confMat = confusionMatrix tp tn fp fn
--         scoF1 = f1score pre re
--         microF1 = microF1score (asValue tp :: [Double]) (asValue fp :: [Double]) (asValue fn :: [Double])
--         macroF1 = macroF1score [scoF1] -- Example with a single F1 score, extend as needed
--         weights = [1.0] -- Example weights, adjust as needed
--         weightedF1 = weightedF1score [scoF1] weights

--     putStrLn ("precision : " ++ show pre)
--     putStrLn ("recall : " ++ show re)
--     putStrLn ("accuracy : " ++ show accu)
--     putStrLn ("confusion matrix : " ++ show confMat)
--     putStrLn ("f1 score : " ++ show scoF1)
--     putStrLn ("micro f1 score : " ++ show microF1)
--     putStrLn ("macro f1 score : " ++ show macroF1)
--     putStrLn ("weighted f1 score : " ++ show weightedF1)
--     return ()