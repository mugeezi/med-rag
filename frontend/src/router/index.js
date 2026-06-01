import { createRouter, createWebHistory } from 'vue-router'
import SearchPage from '../pages/SearchPage.vue'
const routes = [{ path: '/', component: SearchPage }]
export default createRouter({ history: createWebHistory(), routes })
