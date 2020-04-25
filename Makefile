DATA_PATH := data

all: .env

.env :
	@touch $@
	@echo DATA_PATH=$(shell pwd)/$(DATA_PATH) > $@
