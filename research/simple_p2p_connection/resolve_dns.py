import dns.resolver
from multiaddr import Multiaddr

def resolve_dnsaddr_recursive(ma: Multiaddr, depth=0, max_depth=5):
    """
    Recursively resolves a /dnsaddr multiaddr.
    Returns a list of resolved concrete multiaddrs (e.g. starting with /ip4/ or /ip6/).
    """
    if depth > max_depth:
        return [ma]

    protos = ma.protocols()
    if not protos or protos[0].name != "dnsaddr":
        return [ma]

    # Extract the domain from the dnsaddr component
    domain = ma.value_for_protocol(protos[0].code)
    lookup_name = f"_dnsaddr.{domain}"
    resolved = []
    try:
        answers = dns.resolver.resolve(lookup_name, 'TXT')
        for rdata in answers:
            txt = rdata.to_text().strip('"')
            if txt.startswith("dnsaddr="):
                addr_str = txt[len("dnsaddr="):]
                try:
                    resolved_ma = Multiaddr(addr_str)
                    # Check if it still starts with dnsaddr: if so, resolve it recursively.
                    if resolved_ma.protocols()[0].name == "dnsaddr":
                        resolved.extend(resolve_dnsaddr_recursive(resolved_ma, depth+1, max_depth))
                    else:
                        resolved.append(resolved_ma)
                except Exception as e:
                    print(f"Error parsing resolved multiaddr {addr_str}: {e}")
    except Exception as e:
        print(f"DNS TXT lookup failed for {lookup_name}: {e}")
    return resolved

if __name__ == "__main__":
    BOOTSTRAP_NODES = [
        Multiaddr("/dns4/sg1.bootstrap.libp2p.io/tcp/4001/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt"),
        Multiaddr("/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ"),
        Multiaddr("/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN"),
        Multiaddr("/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa"),
        Multiaddr("/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb"),
        Multiaddr("/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt"),
    ]
    
    # Replace with your original dnsaddr bootstrap node
    for bootstrap in BOOTSTRAP_NODES:
        print("Resolving dnsaddr:", bootstrap)
        concrete = resolve_dnsaddr_recursive(bootstrap)
        for addr in concrete:
            print("Resolved concrete address:", addr)
