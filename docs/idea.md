A chat type system with session where theres a reasonable storage . meaning the ai model will have 128k token limit will be using gemini 3.1-flash-lite,  
Upload files in different formats and parsing with markitdown, now for each file which can be very long or short and we can have some context of the file using page index, or chunking summary the best technique available,  
Now the main executor layer will have multiple tools and options

* Search tool using duck duck go ddgs  
* Document parsing tool using mark it down   
* A tool to download any document if link is found which will then be parsed with document parsing tool  
* Python code execution tool which can execute any python code as needed having access to install packages everything  
* crawl4Ai tool without llm layer to fetch html content of any website link  
* Browser-use agent where input will be prompt since it’s prompt based and te task is to extract any information from a website  
* Bm25 agent  
* Gemini search agent for faster discovery of anything  
* Can add more agents later

So the workflow will be based on sessions, based on the user query the agent will create a plan and spawn multiple sub agents as needed to complete the plan  
Now the idea is let’s say the user says to get something from a pdf so it’ll list down what all things to get and use different agent to get it from a document markdown, now after that it might seem that more data is needed to answer the question then update the plan and add more agents and finally return when all plan is done.

Now let’s say the user gives a long prompt where the starting point can be a chat or pdf, now the llm use serach tool to search, if any document found download it, if website found use craws4ai to get the content and then decide what to do next and update plan. It can be like let’s say there’s a regulatory document , then fetch other documents related to that which are linked using search and scraping tool and then finally answers.

So it’s like a full fledged ai harness wiith a lot of power. The more tools that will be added later will contribute to it’s flexibility, 

Now the thing will be like a task can be very long and based on every task design , decide what to do, how to evaluate so that the loop is closed. The prompts need to be proper, the tools needs to be having proper description to make the llm use proper things, as each step happens pass the thinking back to ba basic ui with html js, 

The execution loop can be long or short depending on the user’s requirements. No limit to anything now also since we’ll be doing a lot of iterating and updatin we need to keep track of the context so that the main loop doesn’t run out of it. 

Also it can happen that user force stop the loop and add updated prompt which will redefine what’s done and what needs to be done and what exsiting needs to be kept or replaced

It’ll be multimodal too. If any image is uploaded according to the query use vision llm gemini to get the result and also store a basic description of the image to be used when needed since gemini is multimodal, so it’ll be like a full fledged ai agent for any usecase. 

Also before making any tool or something read their docs using web search and etc things to do a proper thing.

Everything that’s needed do it in memory meaning sqlite, networkx, etc etc