import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";
import "faiss-node";
import "dotenv/config";
import {OllamaEmbeddings} from "@langchain/ollama";

async function run() {
  const directory = "../db/kongyiji";
  // const embeddings = new OpenAIEmbeddings();
   const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // 或 "all-minilm"
  });

  const vectorstore = await FaissStore.load(directory, embeddings);

  const retriever = vectorstore.asRetriever(2);
  const res = await retriever.invoke("茴香豆是做什么用的");

  console.log(res);
}

run();
