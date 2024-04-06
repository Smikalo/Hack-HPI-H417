import 'package:auth_login/provider/route_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class ResultMapScreen extends ConsumerStatefulWidget {
  const ResultMapScreen({
    super.key,
    required this.isSmooth,
  });

  final bool isSmooth;

  @override
  ConsumerState<ResultMapScreen> createState() => _ResultMapScreenState();
}

class _ResultMapScreenState extends ConsumerState<ResultMapScreen> {
  @override
  Widget build(BuildContext context) {
    final address = ref.watch(routeProvider);
    return Scaffold(
        appBar: AppBar(
          title: Text(
            widget.isSmooth ? "Smooth path" : "Quick path",
          ),
        ),
        body: Hero(
            tag: "Map",
            child: FlutterMap(
              options: MapOptions(
                // onTap: (tapPosition, point) {
                //   widget.isCurrent
                //       ? ref
                //           .read(currentAddressProvider.notifier)
                //           .getCurrentAddress(point)
                //       : ref
                //           .read(destinationAddressProvider.notifier)
                //           .getCurrentAddress(point);
                //   getAddressFromCoordinates(point);
                // },
                initialCenter: address.first,
                initialZoom: 11,
                interactionOptions: const InteractionOptions(
                    flags: ~InteractiveFlag.doubleTapZoom),
              ),
              children: [
                TileLayer(
                  urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'dev.fleaflet.flutter_map.example',
                ),
                PolylineLayer(
                  polylines: [
                    Polyline(
                      points: ref.watch(routeProvider),
                      color: Colors.red,
                      strokeWidth: 2,
                    ),
                  ],
                ),
                MarkerLayer(
                  markers: [
                    Marker(
                      height: 50,
                      width: 50,
                      point: address.first,
                      child: const Icon(
                        Icons.location_history,
                        color: Colors.red,
                      ),
                    ),
                    Marker(
                      height: 50,
                      width: 50,
                      point: address.last,
                      child: const Icon(
                        Icons.location_on,
                        color: Colors.red,
                      ),
                    ),
                  ],
                )
              ],
            )));
  }
}
