package utils

import (
	"log"
	"os"
)

var (
	// WarningLogger creates a warning message
	WarningLogger *log.Logger
	// InfoLogger creates an info message
	InfoLogger *log.Logger
	// ErrorLogger creates an error message
	ErrorLogger *log.Logger
)

// Init is initializing the loggers, call it at the beginning of the program, before trying to use the loggers
func Init() {
	InfoLogger = log.New(os.Stdout, "Info: ", log.Ldate|log.Ltime)
	WarningLogger = log.New(os.Stdout, "Warning: ", log.Ldate|log.Ltime)
	ErrorLogger = log.New(os.Stderr, "Error: ", log.Ldate|log.Ltime|log.Lshortfile)
}
