import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

class RouteNotifier extends StateNotifier<List<LatLng>> {
  RouteNotifier() : super(<LatLng>[]);

  void addRoute(List<LatLng> list) {
    state = list;
  }

  void clearRoute() {
    state = [];
  }

  @override
  void dispose() {
    state = [];
    super.dispose();
  }
}

final routeProvider = StateNotifierProvider<RouteNotifier, List<LatLng>>(
  (ref) => RouteNotifier(),
);
