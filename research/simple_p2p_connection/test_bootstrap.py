import trio
import trio_asyncio
from libp2p import new_host
from multiaddr import Multiaddr
from libp2p.peer.peerinfo import info_from_p2p_addr

BOOTSTRAP_NODES = [
    Multiaddr("/ip4/139.178.91.71/tcp/4001/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN"),
    Multiaddr("/ip4/145.40.118.135/tcp/4001/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt"),
    Multiaddr("/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"),
    Multiaddr("/ip4/139.178.65.157/tcp/4001/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa"),
    Multiaddr("/ip4/147.75.87.27/tcp/4001/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb"),
    
]

async def test_bootstrap():
    host = new_host()
    print(f"Testing bootstrap nodes from host: {host.get_id()}")
    for bootstrap in BOOTSTRAP_NODES:
        try:
            bootstrap_info = info_from_p2p_addr(bootstrap)
            await host.connect(bootstrap_info)
            print("Connected to bootstrap node:", bootstrap)
        except Exception as e:
            print("Error connecting to bootstrap node", bootstrap, ":", e)

async def main():
    async with trio_asyncio.open_loop():
        await test_bootstrap()

if __name__ == "__main__":
    trio.run(main)