library(devtools)
#devtools::install_github("scienceverse/papercheck", force =TRUE)

library(scienceverse)
library(papercheck)
library(dplyr)

#install.packages("jsonlite")
library(jsonlite)
library(xml2)
library(stringr)

studies <- read_grobid("All articles XML")

# get text from header participants
specific_header <- "Participants"

count_participants <- 0
empty_partic <- list()

for (i in 1:98){
    #print(studies[[i]])
    headers <- studies[[i]]$full_text$header
    unique_headers <- unique(headers)
    
    if (any(grepl(specific_header, unique_headers))){
        count_participants <- count_participants + 1
        #print(names(studies[[i]]$full_text))

        nameList<-as.numeric(substr(studies[[i]]$full_text$file, 1, 2))
        name <- nameList[1]

        print(name)
        print(i)

        filtered_text <- studies[[i]]$full_text[studies[[i]]$full_text$header == specific_header, ]
        participants <- filtered_text$text

        #chck if section under participants header is empty, if it is mark it such that methods and results can be added to txt
        if (length(participants) > 0) {
            file_name <- paste("text_participants_", name, ".txt", sep = "")
            write(participants, file = file_name)
            print(paste("Paper: ", name, unique_headers[grepl(specific_header, unique_headers)], sep = ""))
        } else {
            message("No participants text found for the specific header: ", specific_header)
            empty_partic[[length(empty_partic) + 1]] <- i
            #print(empty_partic)
        }

        #verification
        #print(paste("Paper: ", name))
        #print(studies[[i]])
    }
    else{
        nameList<-as.numeric(substr(studies[[i]]$full_text$file, 1, 2))
        name <- nameList[1]

        #print(name)
        #print(i)

        text_methods <- paste(search_text(studies[[i]], section="method", return = c("section"))$text, search_text(studies[[i]], section="results", return = c("section"))$text)
        file_name <- paste("text_methods_results", name, ".txt", sep = "")
        write(text_methods, file = file_name)

        #verification
        #print(paste("Paper: ", name))
    }
}

#print list of sections where participant header is found but the section it points to is empty
print('list:')
print(empty_partic)

#add methods and results to txt of these papers
for (i in 1:98){
    headers <- studies[[i]]$full_text$header
    unique_headers <- unique(headers)
    if (any(sapply(empty_partic, function(x) identical(x, i)))){
        nameList<-as.numeric(substr(studies[[i]]$full_text$file, 1, 2))
        name <- nameList[1]

        #print("adding methods for empty txt")
        #print(name)
        #print(i)

        text_methods <- paste(search_text(studies[[i]], section="method", return = c("section"))$text, search_text(studies[[i]], section="results", return = c("section"))$text)
        file_name <- paste("text_methods_results_empty_partic_", name, ".txt", sep = "")
        write(text_methods, file = file_name)

        #verification
        #print(paste("Paper: ", name))
    }
}