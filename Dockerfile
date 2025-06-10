# We base ourselves on Unbuntu. This is the base OS we are installing.
FROM ubuntu:20.04

# Let's CD to a more sensible working directory
WORKDIR /root/workdir

# Copy over the file from your own machine to the container, misspelled
COPY ./README.md ./REAMDE.md