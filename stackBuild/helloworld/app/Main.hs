module Main (main) where
import Data.Char


main :: IO ()
main = do
    putStrLn "What's your first name?"
    firstName <- getLine
    let bigFirstName = map toUpper firstName
    putStrLn $ "hey " ++ bigFirstName ++ ", how are you?"

