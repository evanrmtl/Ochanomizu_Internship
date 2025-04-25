{-# LANGUAGE OverloadedStrings #-}

module Data (
    loadData,
    dataToTensor,
    targetToTensor,
    cgpaToTensor
) where

import Torch.Tensor (Tensor, asTensor)
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import Control.Applicative
import Control.Monad (mzero)

data Admission = Admission
  { serialNo       :: !Int
  , greScore       :: !Int
  , toeflScore     :: !Int
  , universityRating :: !Int
  , sop            :: !Double
  , lor            :: !Double
  , cgpa           :: !Double
  , research       :: !Int
  , chanceOfAdmit  :: !Double
  } deriving Show

instance FromRecord Admission where
  parseRecord v
    | V.length v == 9 = Admission
        <$> v .! 0
        <*> v .! 1
        <*> v .! 2
        <*> v .! 3
        <*> v .! 4
        <*> v .! 5
        <*> v .! 6
        <*> v .! 7
        <*> v .! 8
    | otherwise = mzero

instance FromNamedRecord Admission where
  parseNamedRecord r = Admission
    <$> r .: "Serial No."
    <*> r .: "GRE Score"
    <*> r .: "TOEFL Score"
    <*> r .: "University Rating"
    <*> r .: "SOP"
    <*> r .: "LOR "
    <*> r .: "CGPA"
    <*> r .: "Research"
    <*> r .: "Chance of Admit "

loadData :: FilePath -> IO (Either String (V.Vector Admission))
loadData path = do
  csvData <- BL.readFile path
  case decode NoHeader csvData of
    Left err -> return $ Left ("Erreur de parsing CSV: " ++ err)
    Right v  -> return $ Right v

dataToTensor :: V.Vector Admission -> Tensor
dataToTensor va = asTensor (elements :: [[Float]])
  where
    elements = V.toList $ V.map (\r -> [
      fromIntegral (greScore r) :: Float,
      fromIntegral (toeflScore r) :: Float,
      fromIntegral (universityRating r) :: Float,
      realToFrac (sop r) :: Float,
      realToFrac (lor r) :: Float,
      realToFrac (cgpa r) :: Float,
      fromIntegral (research r) :: Float
      ]) va

targetToTensor :: V.Vector Admission -> Tensor
targetToTensor va = asTensor (elements :: [Float])
  where
    elements = V.toList $ V.map (\r -> realToFrac (chanceOfAdmit r) :: Float) va

cgpaToTensor :: V.Vector Admission -> Tensor
cgpaToTensor va = asTensor (elements :: [Float])
  where
    elements = V.toList $ V.map (\r -> realToFrac (cgpa r) :: Float) va