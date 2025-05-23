This folder is for future development of a data server that a user could connect to to see some shared information. The main goal is to create a p2p platform that would allow to do that seamlessly, but for now the server will still be centralized until the proper protocol for handling content sharing is developed. Although anybody should be able to set up such a server and anybody could connect to any other server, so there still some level of decentralization present. 

Here are some requirements that I would like for Anagnorisis-data-server to have:
* It should be possible to deploy a server by simply running the provided docker container with a specified data folder that will be shared. 
* No static IP or payed domain name should be required. There should not be any additional hurdles that typically come with setting up a traditional server. 
* To protect host and user privacy the IP address of the server should not be visible to the clients and vise verse. 
* It should have some very basic DDOS protection.

Here is some proposal of how such a server might be built:
All the data the host provides is shared via IPFS, while the server itself only provides CID of files + respective metadata that along other information stores an embedding generated by some model. This will allow the server to distribute the load, as only meta information would be shared with the clients directly. If the files are popular enough they might be reshared by other participants, lowering the load on the server.

To ensure the privacy of the host Tor network might be used. The benefit of the Tor network is that it is already widely used, does not disclose current IP address of the host and provides a way to host website without static IP address. This should create a solid foundation that would allow any people, even those who live in countries with strict internet censorship, to host a server. Although an I2P might also be a viable alternative and should be explored as well. 

To protect the server from DDOS attacks, a simple rate limiting might be used, plus Proof-of-Work integration to make attack attempts more resource-intensive.

One concern is that IPFS does not hide the IP address of the host by default, but approaches like IPFS over I2P or Tor exists and might be used to provide additional security.

Another path might be in storing the CID+meta information on some blockchain. This approach might also be explored in the future.