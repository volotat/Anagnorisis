import trio
import trio_asyncio
import asyncio
import sys
import logging

from libp2p import new_host
from multiaddr import Multiaddr
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.pubsub.floodsub import FloodSub
from libp2p.pubsub.pubsub import Pubsub

BOOTSTRAP_NODES = [
    Multiaddr("/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"),
    Multiaddr("/dns4/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN"),
    Multiaddr("/dns4/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa"),
    Multiaddr("/dns4/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb"),
	Multiaddr("/dns4/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt"),
]
CHAT_TOPIC = "p2p-chat-test-gGL5UAdOKM"

async def wait_for_connections(host, timeout=30, interval=2):
    """Poll for connections until at least one peer is connected or until timeout."""
    start = trio.current_time()
    while trio.current_time() - start < timeout:
        # If your libp2p host supports a method to get active connections, use it:
        conns = host.get_connected_peers()  # returns a dict or list of connections
        if conns:
            print("Connections established:", conns)
            return True
        await trio.sleep(interval)
    return False

async def run_chat():
    host = new_host()
    peer_id = host.get_id()
    print(f"Libp2p node started with peer id: {peer_id}")
    print("Listening on addresses:")
    for addr in host.get_addrs():
        print("  ", addr)

    connected_peers = set()
    # Try connecting to bootstrap nodes.
    for bootstrap in BOOTSTRAP_NODES:
        try:
            bootstrap_info = info_from_p2p_addr(bootstrap)
            await host.connect(bootstrap_info)
            connected_peers.add(bootstrap_info.peer_id)
            print("Connected to bootstrap node:", bootstrap)
        except Exception as e:
            print("Error connecting to bootstrap node", bootstrap, ":", e)

    if connected_peers:
        print("Currently connected peers:", ", ".join(connected_peers))
    else:
        print("No immediate bootstrap connections.")
        print("Polling for connections...")
        resulting = await wait_for_connections(host)
        if not resulting:
            print("No connections established within the timeout.")
        else:
            print("Connection(s) established.")

    # Initialize PubSub.
    fsub = FloodSub(protocols=["/floodsub/1.0.0"])
    ps = Pubsub(host, fsub)

    try:
        sub = await ps.subscribe(CHAT_TOPIC)
        print(f"Subscribed to PubSub topic: {CHAT_TOPIC}")
    except Exception as e:
        print("Error subscribing to PubSub topic:", e)
        return

    async def read_incoming():
        while True:
            try:
                msg = await sub.get()
                if msg.from_id != peer_id:
                    print(f"[{msg.from_id}] {msg.data.decode()}")
            except Exception as e:
                print("Error reading incoming message:", e)
                break

    # Only prompt for messaging when connections are active.
    print("Type your messages below and hit enter:")

    async with trio.open_nursery() as nursery:
        nursery.start_soon(read_incoming)
        while True:
            try:
                line = await trio.to_thread.run_sync(sys.stdin.readline)
                line = line.strip()
                if line:
                    await ps.publish(CHAT_TOPIC, line.encode())
            except Exception as e:
                print("Error sending message:", e)

async def main():
    async with trio_asyncio.open_loop():
        await run_chat()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        trio.run(main)
    except KeyboardInterrupt:
        print("Shutting down chat.")