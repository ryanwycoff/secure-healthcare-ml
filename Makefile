# Commands for setting up, testing, and running the application

install:
    pip install -r requirements.txt

run:
    uvicorn api.main:app --reload

build:
    docker-compose up --build

test:
    pytest tests/
